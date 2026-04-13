import socket
import struct
import time
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import threading
import json
from dataclasses import dataclass, field
from openai import OpenAI

# ==========================================
# 0. OpenAI Client
# ==========================================
# 需先設定環境變數 OPENAI_API_KEY
# Windows PowerShell:
# $env:OPENAI_API_KEY="你的_api_key"
client = OpenAI()

# ==========================================
# 1. 參數設定
# ==========================================
OMEGA_ALT = np.array([[11.2683, 16.5707], [16.5707, 25.0074]], dtype=float)
F1_ALT = np.array([-8.1979, -12.7244], dtype=float)
SIGMA_ALT = 0.05
AR_ALT = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)
HOVER_THRUST = 0.72
THRUST_SCALE = 0.045

OMEGA_POS = np.array([[282.0784, 501.9677], [501.9677, 893.5577]], dtype=float)
F1_POS = np.array([-2.4604, -4.4041], dtype=float)
F2_POS = np.array([-2.9273, -5.2102], dtype=float)
AR_POS = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)

SIGMA_POS = 0.40
POS_ERR_MAX = 1.0
RATE_LIMIT_XY = 3.0
GAIN_SCALE_XY = 0.06
MIN_TRIGGER_INTERVAL = 0.05
TAU_FILTER_XY = 0.15
POS_DEADZONE_XY = 0.03
VEL_DEADZONE_XY = 0.05
SOFT_ERR_XY = 0.5
ALPHA_MIN = 0.15
FF_ACCEL_GAIN = 0.102
ROLL_PITCH_LIMIT = 0.35
DT_MIN = 0.01
DT_MAX = 0.03

TARGET_Z = 5.0
LANDING_SPEED = 0.5

XY_ENABLE_ALTITUDE = 0.30
LOW_ALT_XY_GAIN = 0.30
LOW_ALT_ROLL_PITCH_LIMIT = 0.03
THR_MIN = 0.30
THR_MAX = 0.90
ALT_SOFT_CEIL = 6.0
ALT_HARD_CEIL = 8.0

# 已改成待命模式，不自動跑 figure-8
ENABLE_FIGURE8 = False
HOVER_BEFORE_TRAJ = 8.0
FIG8_A = 18.0
FIG8_B = 10.0
FIG8_OMEGA = 0.05
FIG8_PERIOD = 2.0 * math.pi / FIG8_OMEGA
FIG8_LOOPS = 1
TOTAL_FIG8_TIME = FIG8_PERIOD * FIG8_LOOPS

# ==========================================
# 1-0. 顯示/CLI 設定
# ==========================================
PRINT_ENABLE = False   # 先關掉，避免洗版；要看狀態再改 True
PRINT_PERIOD = 2.0
CLI_TYPING = False

# ==========================================
# 1-1. Policy 設定
# ==========================================
POLICY_CONFIG = {
    "normal": {
        "xy_scale": 1.00,
        "ff_scale": 1.00,
        "rp_limit_scale": 1.00,
        "reach_tol": 0.35,
    },
    "energy_saving": {
        "xy_scale": 0.65,
        "ff_scale": 0.55,
        "rp_limit_scale": 0.70,
        "reach_tol": 0.45,
    },
    "smooth": {
        "xy_scale": 0.75,
        "ff_scale": 0.70,
        "rp_limit_scale": 0.75,
        "reach_tol": 0.40,
    },
    "fast": {
        "xy_scale": 1.20,
        "ff_scale": 1.15,
        "rp_limit_scale": 1.15,
        "reach_tol": 0.30,
    },
    "safe": {
        "xy_scale": 0.55,
        "ff_scale": 0.50,
        "rp_limit_scale": 0.60,
        "reach_tol": 0.50,
    },
}
VALID_POLICIES = set(POLICY_CONFIG.keys())

# ==========================================
# 2. GAI / 任務層狀態
# ==========================================
MISSION_LOCK = threading.Lock()


@dataclass
class MissionState:
    mode: str = "auto_traj"   # auto_traj / hold / goto / inspect / rth / traj
    frame: str = "local"      # local / body / none
    policy: str = "normal"    # normal / energy_saving / smooth / fast / safe

    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = TARGET_Z
    target_yaw: float = 0.0

    path: list = field(default_factory=list)
    path_index: int = 0

    hold_time: float = 0.0
    hold_start_time: float = 0.0
    
    traj_type: str = ""
    traj_param: float = 10.0

    active: bool = False
    source_text: str = ""
    last_update_time: float = field(default_factory=time.time)


mission_state = MissionState()

mission_runtime_state = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "yaw": 0.0,
    "home_x": 0.0,
    "home_y": 0.0,
}

# ==========================================
# 3. Z 軸與 XY 軸控制器
# ==========================================
class Altitude_ETM_Controller:
    """
    無人機高度 (Z軸) 的事件觸發機制 (ETM) 控制器。
    基於給定目標高度計算追蹤誤差，並在網路通訊誤差超過門檻才觸發新的控制指令傳送，
    以減少不必要的通訊負載。
    """
    def __init__(self):
        self.last_sent_state = np.zeros(2, dtype=float)  # 上次傳送的高度狀態 [z, vz]
        self.last_control_u = 0.0                       # 上次計算的控制輸出
        self.ref_state = np.zeros(2, dtype=float)       # 參考模型狀態
        self.prev_time = time.time()
        self.ref_state[0] = 0.0
        self.first_run = True

    def update_reference(self, dt, target_height):
        # 計算參考模型演進，產生平滑的高度追蹤曲線
        r_input = np.array([0.0, 4.0 * target_height], dtype=float)
        dx_r = AR_ALT @ self.ref_state + r_input
        self.ref_state += dx_r * dt
        return self.ref_state

    def compute_control(self, current_state, target_height):
        # 根據系統當前狀態計算 ETM 控制輸出，並返回是否觸發事件的布林值
        current_time = time.time()
        dt = current_time - self.prev_time
        dt = float(np.clip(dt, DT_MIN, DT_MAX))
        self.prev_time = current_time

        xr = self.update_reference(dt, target_height)
        e_trk = current_state - xr                      # 追蹤誤差
        e_net = self.last_sent_state - current_state      # 網路狀態偏差

        term_net = float(e_net.T @ OMEGA_ALT @ e_net)
        term_trk = float(e_trk.T @ OMEGA_ALT @ e_trk)

        triggered = False
        # 判斷是否滿足事件觸發條件
        if self.first_run or (term_net > SIGMA_ALT * term_trk):
            triggered = True
            self.first_run = False
            self.last_sent_state = current_state.copy()
            u = float(F1_ALT @ e_trk)
            self.last_control_u = u
        else:
            u = self.last_control_u

        return u, triggered


class Fuzzy_ETM_Core:
    """
    無人機水平控制 (X 與 Y 軸) 的模糊事件觸發核心機制。
    結合死區(deadzone)與軟變數(soft_scale)，以及模糊增益表 (w1*F1 + w2*F2)，
    在減少通訊次數的同時改善誤差震盪問題。
    """
    def __init__(
        self,
        omega,
        f1,
        f2,
        ar,
        name="Sys",
        rate_limit=1.5,
        gain_scale=1.0,
        pos_deadzone=0.03,
        vel_deadzone=0.05,
        soft_err=0.3,
    ):
        self.Omega = omega
        self.F1 = f1
        self.F2 = f2
        self.Ar = ar
        self.name = name
        self.rate_limit = float(rate_limit)
        self.gain_scale = float(gain_scale)
        self.pos_deadzone = float(pos_deadzone)
        self.vel_deadzone = float(vel_deadzone)
        self.soft_err = float(soft_err)

        self.last_sent_error = np.zeros(2, dtype=float)
        self.filtered_error = np.zeros(2, dtype=float)
        self.last_control_u = 0.0
        self.ref_state = np.zeros(2, dtype=float)
        self.prev_final_u = 0.0
        self.first_run = True
        self.last_trigger_time = 0.0

    def get_fuzzy_gain(self, e_trk):
        # 根據誤差值內插不同增益矩陣以調整響應特性
        abs_err = abs(float(e_trk[0]))
        w2 = float(np.clip(abs_err / POS_ERR_MAX, 0.0, 1.0))
        w1 = 1.0 - w2
        F_fuzzy = w1 * self.F1 + w2 * self.F2
        return F_fuzzy

    def apply_deadzone(self, e):
        # 微小誤差不施加控制以消除穩態震盪
        e_out = e.copy()
        if abs(e_out[0]) < self.pos_deadzone:
            e_out[0] = 0.0
        if abs(e_out[1]) < self.vel_deadzone:
            e_out[1] = 0.0
        return e_out

    def soft_scale(self, e):
        # 對大誤差給予比例保護限制
        e_norm = float(np.linalg.norm(e))
        alpha = min(1.0, e_norm / self.soft_err) if self.soft_err > 1e-9 else 1.0
        alpha = max(alpha, ALPHA_MIN)
        return alpha

    def update(self, current_state, target_val, dt):
        """
        執行運算以生成 X 或 Y 軸的 pitch/roll 控制輸入。
        根據目前狀態 (current_state) 和目標變量 (target_val)，判斷是否觸發(trigger)。
        """
        dt = float(np.clip(dt, DT_MIN, DT_MAX))
        now = time.time()

        # 生成本軸平滑參考線
        r_input = np.array([0.0, 4.0 * target_val], dtype=float)
        dx_r = self.Ar @ self.ref_state + r_input
        self.ref_state += dx_r * dt

        e_trk = current_state - self.ref_state    # 軌跡誤差
        e_net = self.last_sent_error - e_trk      # 網路延遲造成的傳輸與計算偏差

        term_net = float(e_net.T @ self.Omega @ e_net)
        term_trk = float(e_trk.T @ self.Omega @ e_trk)

        triggered = False
        # 水平誤差觸發條件 (附加最短觸發間隔限制，避免過密網路負載)
        if self.first_run or (
            (term_net > SIGMA_POS * term_trk)
            and ((now - self.last_trigger_time) >= MIN_TRIGGER_INTERVAL)
        ):
            triggered = True
            self.first_run = False
            self.last_trigger_time = now
            self.last_sent_error = e_trk.copy()

        # 使用低通濾波來過濾雜訊，避免干擾抖動
        self.filtered_error += (dt / TAU_FILTER_XY) * (
            self.last_sent_error - self.filtered_error
        )
        e_ctrl = self.apply_deadzone(self.filtered_error)

        # 計算模糊控制增益與非線性軟限制
        F_fuzzy = self.get_fuzzy_gain(e_ctrl)
        alpha = self.soft_scale(e_ctrl)

        # 控制輸入限幅與平滑防震率限制
        u_raw = float(alpha * (F_fuzzy @ e_ctrl) * self.gain_scale)
        du = (u_raw - self.prev_final_u) / dt
        if abs(du) > self.rate_limit:
            u_final = self.prev_final_u + np.sign(du) * self.rate_limit * dt
        else:
            u_final = u_raw

        self.last_control_u = float(u_final)
        self.prev_final_u = self.last_control_u

        return self.last_control_u, triggered

# ==========================================
# 4. 任務工具函式
# ==========================================
def clip_mission_target(x, y, z, home_x, home_y):
    max_xy = 30.0
    min_z = XY_ENABLE_ALTITUDE
    max_z = min(TARGET_Z + 5.0, ALT_HARD_CEIL)  # 放寬至最大容許值

    x = float(np.clip(x, home_x - max_xy, home_x + max_xy))
    y = float(np.clip(y, home_y - max_xy, home_y + max_xy))
    z = float(np.clip(z, min_z, max_z))
    return x, y, z


def body_to_world(dx, dy, yaw):
    wx = math.cos(yaw) * dx - math.sin(yaw) * dy
    wy = math.sin(yaw) * dx + math.cos(yaw) * dy
    return wx, wy


def normalize_policy(policy):
    if policy in VALID_POLICIES:
        return policy
    return "normal"


def set_hold_mission(x, y, z, yaw=0.0, hold_time=0.0, source_text="", policy="normal"):
    with MISSION_LOCK:
        mission_state.mode = "hold"
        mission_state.frame = "local"
        mission_state.policy = normalize_policy(policy)
        mission_state.target_x = x
        mission_state.target_y = y
        mission_state.target_z = z
        mission_state.target_yaw = yaw
        mission_state.hold_time = hold_time
        mission_state.hold_start_time = time.time()
        mission_state.active = True
        mission_state.source_text = source_text
        mission_state.last_update_time = time.time()


def set_path_mission(waypoints, source_text="", policy="normal"):
    with MISSION_LOCK:
        mission_state.mode = "follow_path"
        mission_state.frame = "local"
        mission_state.policy = normalize_policy(policy)
        mission_state.path = waypoints
        mission_state.path_index = 0
        
        if len(waypoints) > 0:
            mission_state.target_x = waypoints[0]["x"]
            mission_state.target_y = waypoints[0]["y"]
            mission_state.target_z = waypoints[0]["z"]
            mission_state.target_yaw = waypoints[0]["yaw"]
            
        mission_state.hold_time = 0.0
        mission_state.hold_start_time = 0.0
        mission_state.active = True
        mission_state.source_text = source_text
        mission_state.last_update_time = time.time()


def set_goto_mission(x, y, z, yaw=0.0, source_text="", policy="normal"):
    with MISSION_LOCK:
        mission_state.mode = "goto"
        mission_state.frame = "local"
        mission_state.policy = normalize_policy(policy)
        mission_state.target_x = x
        mission_state.target_y = y
        mission_state.target_z = z
        mission_state.target_yaw = yaw
        mission_state.hold_time = 0.0
        mission_state.hold_start_time = 0.0
        mission_state.active = True
        mission_state.source_text = source_text
        mission_state.last_update_time = time.time()


def set_inspect_mission(x, y, z, hold_time=5.0, yaw=0.0, source_text="", policy="normal"):
    with MISSION_LOCK:
        mission_state.mode = "inspect"
        mission_state.frame = "local"
        mission_state.policy = normalize_policy(policy)
        mission_state.target_x = x
        mission_state.target_y = y
        mission_state.target_z = z
        mission_state.target_yaw = yaw
        mission_state.hold_time = hold_time
        mission_state.hold_start_time = 0.0
        mission_state.active = True
        mission_state.source_text = source_text
        mission_state.last_update_time = time.time()


def set_rth_mission(source_text="", policy="safe"):
    with MISSION_LOCK:
        mission_state.mode = "rth"
        mission_state.frame = "local"
        mission_state.policy = normalize_policy(policy)
        mission_state.active = True
        mission_state.source_text = source_text
        mission_state.last_update_time = time.time()


def clear_mission():
    with MISSION_LOCK:
        mission_state.mode = "auto_traj"
        mission_state.frame = "local"
        mission_state.policy = "normal"
        mission_state.active = False
        mission_state.hold_time = 0.0
        mission_state.hold_start_time = 0.0
        mission_state.path = []
        mission_state.path_index = 0
        mission_state.source_text = ""
        mission_state.last_update_time = time.time()

# ==========================================
# 5. OpenAI 任務解析
# ==========================================
def parse_nl_command_with_openai(user_text, current_state, home_state, image_path=None):
    """
    透過 OpenAI API 將口語化的中文指令（如"升空"、"往南飛兩米"、"幫我跑個八字形"），
    解析為系統可直接讀取的結構化 JSON 任務描述。
    若提供 image_path，則結合多模態分析結果。
    """
    schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["go_to", "hover", "relative_move", "inspect", "run_trajectory", "return_home", "reject", "follow_path"],
            },
            "frame": {
                "type": "string",
                "enum": ["local", "body", "none"],
            },
            "path": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "yaw": {"type": "number"},
                    },
                    "required": ["x", "y", "z", "yaw"],
                    "additionalProperties": False,
                }
            },
            "target": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "yaw": {"type": "number"},
                },
                "required": ["x", "y", "z", "yaw"],
                "additionalProperties": False,
            },
            "offset": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "yaw": {"type": "number"},
                },
                "required": ["x", "y", "z", "yaw"],
                "additionalProperties": False,
            },
            "traj_type": {
                "type": "string",
                "enum": ["fig8", "circle", ""]
            },
            "hold_time": {"type": "number"},
            "policy": {
                "type": "string",
                "enum": ["normal", "energy_saving", "smooth", "fast", "safe"],
            },
            "reason": {"type": "string"},
        },
        "required": ["command", "frame", "target", "offset", "path", "traj_type", "hold_time", "policy", "reason"],
        "additionalProperties": False,
    }

    prompt = {
        "instruction": "將中文或附加的無人機任務轉成單一 JSON。若包含圖片則須以圖片判斷意圖。",
        "rules": [
            "不可輸出 roll pitch thrust pwm motor",
            "若圖片中可辨識出一段飛行路徑、包含多個檢查點，令 command=follow_path 並將座標序列放入 path 陣列中 (須考慮從目前位置合理推進)",
            "若要求執行八字形或圓形飛行軌跡，使用 command=run_trajectory，且 traj_type=fig8 或 circle",
            "若語意不夠清楚或有風險，command=reject",
            "前後左右上下等相對描述用 body frame 且為 relative_move",
            "往上飛 z 為正，往下飛 z 為負。例如往上飛三公尺 offset.z=3.0",
            "明確 x y z 座標用 local frame",
            f"絕對高度 z 必須考慮限制，最高不要超過 {ALT_HARD_CEIL:.2f} 公尺",
            "yaw 沒指定就給 0.0",
            "hold_time 沒指定就給 0.0",
            "若使用者提到節能、省電，policy=energy_saving",
            "若使用者提到平穩、平滑、不要太激烈，policy=smooth",
            "若使用者提到最快、快速，policy=fast",
            "若使用者提到安全、保守，policy=safe",
            "若沒有特別偏好，policy=normal",
        ],
        "current_state": current_state,
        "home_state": home_state,
        "user_text": user_text,
    }

    messages = [{"role": "system", "content": "你是無人機任務命令解析器，只能輸出符合 schema 的 JSON。"}]
    
    if image_path:
        import base64
        try:
            with open(image_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "加上參考這張圖片。 " + json.dumps(prompt, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            })
        except Exception as e:
            print(f"[OpenAI Error] 無法讀取圖片 {e}")
            messages.append({"role": "user", "content": json.dumps(prompt, ensure_ascii=False)})
    else:
        messages.append({"role": "user", "content": json.dumps(prompt, ensure_ascii=False)})

    import traceback
    try:
        # 兼容 responses API 或標準 chat API
        if hasattr(client, "responses"):
            response = client.responses.create(
                model="gpt-5.4" if image_path else "gpt-4o",  # 有圖片用 5.4，沒圖片用 4o
                input=messages,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "uav_command",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            return json.loads(response.output_text)
        else:
            response = client.chat.completions.create(
                model="gpt-5.4" if image_path else "gpt-4o",
                messages=messages,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI parsing failed: {e}")
        traceback.print_exc()
        return {}

def validate_parsed_command(cmd):
    command = cmd.get("command", "")
    frame = cmd.get("frame", "none")
    target = cmd.get("target", {})
    offset = cmd.get("offset", {})
    path = cmd.get("path", [])
    hold_time = float(cmd.get("hold_time", 0.0))
    policy = cmd.get("policy", "normal")
    traj_type = cmd.get("traj_type", "")

    if command not in {"go_to", "hover", "relative_move", "inspect", "run_trajectory", "return_home", "reject", "follow_path"}:
        return False, "unsupported command"

    if frame not in {"local", "body", "none"}:
        return False, "unsupported frame"

    if policy not in VALID_POLICIES:
        return False, "unsupported policy"

    if hold_time < 0.0 or hold_time > 60.0:
        return False, "invalid hold_time"

    if command in {"go_to", "inspect"}:
        z = float(target.get("z", 0.0))
        if not (XY_ENABLE_ALTITUDE <= z <= ALT_HARD_CEIL):
            return False, "target z out of range"

    if command == "follow_path":
        if not path or len(path) == 0:
            return False, "missing path waypoints"
        for wp in path:
            z = float(wp.get("z", TARGET_Z))
            if not (XY_ENABLE_ALTITUDE <= z <= ALT_HARD_CEIL):
                return False, "a waypoint z is out of range"

    if command == "relative_move":
        dx = abs(float(offset.get("x", 0.0)))
        dy = abs(float(offset.get("y", 0.0)))
        dz = abs(float(offset.get("z", 0.0)))
        if dx > 30.0 or dy > 30.0 or dz > 10.0:
            return False, "relative move too large"
            
    if command == "run_trajectory":
        if traj_type not in {"fig8", "circle"}:
            return False, "unsupported trajectory type"

    return True, ""


def apply_parsed_command(cmd, current_state, home_state):
    ok, reason = validate_parsed_command(cmd)
    if not ok:
        print(f"[Mission] Reject by validator: {reason}")
        return

    command = cmd["command"]
    frame = cmd["frame"]
    target = cmd["target"]
    offset = cmd["offset"]
    path_data = cmd.get("path", [])
    hold_time = float(cmd["hold_time"])
    policy = cmd.get("policy", "normal")
    traj_type = cmd.get("traj_type", "")

    x = current_state["x"]
    y = current_state["y"]
    z = current_state["z"]
    yaw = current_state["yaw"]

    if command == "reject":
        print(f"[Mission] LLM rejected: {cmd.get('reason', '')}")
        return

    if command == "run_trajectory":
        with MISSION_LOCK:
            mission_state.mode = "traj"
            mission_state.traj_type = traj_type
            mission_state.traj_param = 10.0  # 預設參數
            mission_state.policy = policy
            mission_state.active = True
            mission_state.hold_start_time = time.time()
        print(f"[Mission] TRAJ {traj_type} from LLM | policy={policy}")
        return

    if command == "hover":
        tx, ty, tz = clip_mission_target(x, y, z, home_state["x"], home_state["y"])
        set_hold_mission(tx, ty, tz, yaw, hold_time, source_text="hover", policy=policy)
        print(f"[Mission] HOLD at ({tx:.2f}, {ty:.2f}, {tz:.2f}) for {hold_time:.1f}s | policy={policy}")
        return

    if command == "return_home":
        tx, ty, tz = clip_mission_target(
            home_state["x"], home_state["y"], TARGET_Z, home_state["x"], home_state["y"]
        )
        set_goto_mission(tx, ty, tz, 0.0, source_text="return_home", policy=policy)
        print(f"[Mission] RTH to ({tx:.2f}, {ty:.2f}, {tz:.2f}) | policy={policy}")
        return

    if command == "go_to":
        if frame == "body":
            dx, dy = body_to_world(float(target["x"]), float(target["y"]), yaw)
            tx = x + dx
            ty = y + dy
            tz = float(target["z"])
        else:
            tx = float(target["x"])
            ty = float(target["y"])
            tz = float(target["z"])

        tx, ty, tz = clip_mission_target(tx, ty, tz, home_state["x"], home_state["y"])
        set_goto_mission(tx, ty, tz, float(target["yaw"]), source_text="go_to", policy=policy)
        print(f"[Mission] GOTO ({tx:.2f}, {ty:.2f}, {tz:.2f}) | policy={policy}")
        return

    if command == "follow_path":
        pts = []
        for wp in path_data:
            wx, wy, wz = clip_mission_target(
                float(wp.get("x", 0)), float(wp.get("y", 0)), float(wp.get("z", TARGET_Z)),
                home_state["x"], home_state["y"]
            )
            pts.append({
                "x": wx, "y": wy, "z": wz, "yaw": float(wp.get("yaw", 0.0))
            })
        set_path_mission(pts, source_text="follow_path", policy=policy)
        if pts:
            print(f"[Mission] FOLLOW_PATH with {len(pts)} points, moving to first: ({pts[0]['x']:.2f}, {pts[0]['y']:.2f}) | policy={policy}")
        return

    if command == "relative_move":
        dx_body = float(offset["x"])
        dy_body = float(offset["y"])
        dz = float(offset["z"])
        dx, dy = body_to_world(dx_body, dy_body, yaw)
        tx = x + dx
        ty = y + dy
        tz = z + dz
        tx, ty, tz = clip_mission_target(tx, ty, tz, home_state["x"], home_state["y"])
        set_goto_mission(tx, ty, tz, yaw + float(offset["yaw"]), source_text="relative_move", policy=policy)
        print(f"[Mission] REL_MOVE -> ({tx:.2f}, {ty:.2f}, {tz:.2f}) | policy={policy}")
        return

    if command == "inspect":
        if frame == "body":
            dx, dy = body_to_world(float(target["x"]), float(target["y"]), yaw)
            tx = x + dx
            ty = y + dy
            tz = float(target["z"])
        else:
            tx = float(target["x"])
            ty = float(target["y"])
            tz = float(target["z"])

        tx, ty, tz = clip_mission_target(tx, ty, tz, home_state["x"], home_state["y"])
        use_hold = hold_time if hold_time > 0.0 else 5.0
        set_inspect_mission(
            tx,
            ty,
            tz,
            hold_time=use_hold,
            yaw=float(target["yaw"]),
            source_text="inspect",
            policy=policy,
        )
        print(f"[Mission] INSPECT ({tx:.2f}, {ty:.2f}, {tz:.2f}) hold {use_hold:.1f}s | policy={policy}")
        return

# ==========================================
# 6. 命令列執行緒
# ==========================================
def command_thread_fn():
    global CLI_TYPING

    print("\n[Mission CLI] 可輸入：")
    print("  auto")
    print("  hover")
    print("  go x y z [policy]")
    print("  move dx dy dz [policy]")
    print("  inspect x y z hold [policy]")
    print("  traj fig8/circle [param] [policy]")
    print("  rth [policy]")
    print("  nl 你的中文命令")
    print("  voice <audio_file_path>")
    print("  mic [錄音秒數=5]")
    print("  img <image_file_path> [你的中文命令]")
    print("  q\n")

    while True:
        try:
            CLI_TYPING = True
            cmdline = input("[Mission CLI] > ").strip()
        except EOFError:
            break
        finally:
            CLI_TYPING = False

        if not cmdline:
            continue

        if cmdline.lower() in {"q", "quit", "exit"}:
            break

        try:
            with MISSION_LOCK:
                curr = {
                    "x": mission_runtime_state["x"],
                    "y": mission_runtime_state["y"],
                    "z": mission_runtime_state["z"],
                    "yaw": mission_runtime_state["yaw"],
                }
                home = {
                    "x": mission_runtime_state["home_x"],
                    "y": mission_runtime_state["home_y"],
                    "z": TARGET_Z,
                }

            parts = cmdline.split()

            if parts[0] == "auto":
                clear_mission()
                print("[Mission] back to auto trajectory")
                continue

            if parts[0] == "hover":
                set_hold_mission(curr["x"], curr["y"], curr["z"], curr["yaw"], 0.0, "hover", policy="normal")
                print("[Mission] HOLD current position | policy=normal")
                continue

            if parts[0] == "go" and len(parts) in {4, 5}:
                tx, ty, tz = map(float, parts[1:4])
                policy = normalize_policy(parts[4]) if len(parts) == 5 else "normal"
                tx, ty, tz = clip_mission_target(tx, ty, tz, home["x"], home["y"])
                set_goto_mission(tx, ty, tz, 0.0, "manual_go", policy=policy)
                print(f"[Mission] GOTO ({tx:.2f}, {ty:.2f}, {tz:.2f}) | policy={policy}")
                continue

            if parts[0] == "move" and len(parts) in {4, 5}:
                dx, dy, dz = map(float, parts[1:4])
                policy = normalize_policy(parts[4]) if len(parts) == 5 else "normal"
                wx, wy = body_to_world(dx, dy, curr["yaw"])
                tx, ty, tz = clip_mission_target(
                    curr["x"] + wx,
                    curr["y"] + wy,
                    curr["z"] + dz,
                    home["x"],
                    home["y"],
                )
                set_goto_mission(tx, ty, tz, curr["yaw"], "manual_move", policy=policy)
                print(f"[Mission] REL_MOVE -> ({tx:.2f}, {ty:.2f}, {tz:.2f}) | policy={policy}")
                continue

            if parts[0] == "inspect" and len(parts) in {5, 6}:
                tx, ty, tz, hold = map(float, parts[1:5])
                policy = normalize_policy(parts[5]) if len(parts) == 6 else "normal"
                tx, ty, tz = clip_mission_target(tx, ty, tz, home["x"], home["y"])
                set_inspect_mission(tx, ty, tz, hold, 0.0, "manual_inspect", policy=policy)
                print(f"[Mission] INSPECT ({tx:.2f}, {ty:.2f}, {tz:.2f}) hold {hold:.1f}s | policy={policy}")
                continue

            if parts[0] == "rth":
                policy = normalize_policy(parts[1]) if len(parts) == 2 else "safe"
                set_rth_mission("manual_rth", policy=policy)
                print(f"[Mission] RTH activated | policy={policy}")
                continue

            if parts[0] == "traj" and len(parts) >= 2:
                traj_type = parts[1]
                param = float(parts[2]) if len(parts) >= 3 else 10.0
                policy = normalize_policy(parts[3]) if len(parts) >= 4 else "normal"
                
                with MISSION_LOCK:
                    mission_state.mode = "traj"
                    mission_state.traj_type = traj_type
                    mission_state.traj_param = param
                    mission_state.policy = policy
                    mission_state.active = True
                    mission_state.hold_start_time = time.time()
                print(f"[Mission] TRAJ {traj_type} param={param} | policy={policy}")
                continue

            if parts[0] == "nl":
                nl_text = cmdline[3:].strip()
                if not nl_text:
                    print("[Mission] empty NL text")
                    continue

                parsed = parse_nl_command_with_openai(nl_text, curr, home)
                print("[Mission][LLM JSON]", json.dumps(parsed, ensure_ascii=False))
                apply_parsed_command(parsed, curr, home)
                continue

            if parts[0] == "voice":
                if len(parts) < 2:
                    print("[Mission] missing audio file path")
                    continue
                audio_path = parts[1]
                try:
                    with open(audio_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="gpt-5.4", 
                            file=audio_file
                        )
                    text = transcript.text
                    print(f"[Voice] 語音辨識結果: {text}")
                    parsed = parse_nl_command_with_openai(text, curr, home)
                    print("[Mission][LLM JSON]", json.dumps(parsed, ensure_ascii=False))
                    apply_parsed_command(parsed, curr, home)
                except Exception as e:
                    print(f"[Mission] Voice error: {e}")
                continue

            if parts[0] == "mic":
                dur = float(parts[1]) if len(parts) >= 2 else 5.0
                print(f"[Mic] 準備錄音，請說話 ({dur} 秒)...")
                try:
                    import sounddevice as sd
                    import scipy.io.wavfile as wavfile
                    
                    fs = 44100
                    recording = sd.rec(int(dur * fs), samplerate=fs, channels=1, dtype='int16')
                    sd.wait()
                    wav_path = "temp_mic_record.wav"
                    wavfile.write(wav_path, fs, recording)
                    print("[Mic] 錄音結束，正在辨識...")
                    
                    with open(wav_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="gpt-5.4", 
                            file=audio_file
                        )
                    text = transcript.text
                    print(f"[Mic] 語音辨識結果: {text}")
                    
                    parsed = parse_nl_command_with_openai(text, curr, home)
                    print("[Mission][LLM JSON]", json.dumps(parsed, ensure_ascii=False))
                    apply_parsed_command(parsed, curr, home)
                except ImportError:
                    print("[Mic] 缺少錄音相關套件，請在終端機安裝: pip install sounddevice scipy")
                except Exception as e:
                    print(f"[Mission] Mic error: {e}")
                continue

            if parts[0] == "img":
                if len(parts) < 2:
                    print("[Mission] missing image file path")
                    continue
                img_path = parts[1]
                nl_text = " ".join(parts[2:]) if len(parts) > 2 else "請根據圖片指派任務"
                try:
                    parsed = parse_nl_command_with_openai(nl_text, curr, home, image_path=img_path)
                    print("[Mission][LLM JSON]", json.dumps(parsed, ensure_ascii=False))
                    apply_parsed_command(parsed, curr, home)
                except Exception as e:
                    print(f"[Mission] Image error: {e}")
                continue

            print("[Mission] unsupported command")

        except Exception as e:
            print(f"[Mission] error: {e}")

# ==========================================
# 7. 軌跡狀態機（含任務覆蓋）
# ==========================================
def generate_trajectory(home_x, home_y, elapsed, current_x, current_y, current_z, current_yaw):
    """
    軌跡生成與任務狀態機：
    根據目前的無人機狀態 (MissionState) 與任務類型(hold, goto, inspect, traj, rth等)，
    決定下一瞬間無人機所應當抵達的目標位置 [tx, ty, tz] 或速度/加速度 [vx, vy, ax, ay] 。
    其中 `traj` 模式支援了即時推算動態的「八字」與「圓形」飛行方程式。
    """
    with MISSION_LOCK:
        ms = MissionState(**mission_state.__dict__)

    policy_cfg = POLICY_CONFIG.get(ms.policy, POLICY_CONFIG["normal"])
    reach_tol = policy_cfg["reach_tol"]

    if ms.active:
        if ms.mode == "hold":
            if ms.hold_time > 0.0:
                if (time.time() - ms.hold_start_time) >= ms.hold_time:
                    clear_mission()
                    return current_x, current_y, current_z, 0.0, 0.0, 0.0, 0.0, "idle"
            return ms.target_x, ms.target_y, ms.target_z, 0.0, 0.0, 0.0, 0.0, "mission_hold"

        elif ms.mode == "goto":
            dist = math.sqrt(
                (current_x - ms.target_x) ** 2
                + (current_y - ms.target_y) ** 2
                + (current_z - ms.target_z) ** 2
            )
            if dist < reach_tol:
                clear_mission()
                return current_x, current_y, current_z, 0.0, 0.0, 0.0, 0.0, "idle"
            return ms.target_x, ms.target_y, ms.target_z, 0.0, 0.0, 0.0, 0.0, "mission_goto"

        elif ms.mode == "follow_path":
            dist = math.sqrt(
                (current_x - ms.target_x) ** 2
                + (current_y - ms.target_y) ** 2
                + (current_z - ms.target_z) ** 2
            )
            if dist < reach_tol:
                # 抵達當前航點，切換至下一個
                with MISSION_LOCK:
                    ms.path_index += 1
                    if ms.path_index < len(ms.path):
                        mission_state.path_index = ms.path_index
                        mission_state.target_x = ms.path[ms.path_index]["x"]
                        mission_state.target_y = ms.path[ms.path_index]["y"]
                        mission_state.target_z = ms.path[ms.path_index]["z"]
                        mission_state.target_yaw = ms.path[ms.path_index]["yaw"]
                        print(f"[Mission] WP Reached. Next: ({mission_state.target_x:.2f}, {mission_state.target_y:.2f})")
                        return mission_state.target_x, mission_state.target_y, mission_state.target_z, 0.0, 0.0, 0.0, 0.0, "mission_path"
                    else:
                        print("[Mission] Path Finished.")
                        clear_mission()
                        return current_x, current_y, current_z, 0.0, 0.0, 0.0, 0.0, "idle"
            return ms.target_x, ms.target_y, ms.target_z, 0.0, 0.0, 0.0, 0.0, "mission_path"

        elif ms.mode == "inspect":
            dist = math.sqrt(
                (current_x - ms.target_x) ** 2
                + (current_y - ms.target_y) ** 2
                + (current_z - ms.target_z) ** 2
            )
            if dist < reach_tol:
                if ms.hold_start_time <= 0.0:
                    with MISSION_LOCK:
                        mission_state.hold_start_time = time.time()
                else:
                    if (time.time() - mission_state.hold_start_time) >= ms.hold_time:
                        clear_mission()
                        return current_x, current_y, current_z, 0.0, 0.0, 0.0, 0.0, "idle"
            return ms.target_x, ms.target_y, ms.target_z, 0.0, 0.0, 0.0, 0.0, "mission_inspect"

        elif ms.mode == "rth":
            tx, ty, tz = home_x, home_y, TARGET_Z
            dist = math.sqrt(
                (current_x - tx) ** 2
                + (current_y - ty) ** 2
                + (current_z - tz) ** 2
            )
            if dist < reach_tol:
                set_hold_mission(tx, ty, tz, 0.0, 0.0, "rth_hold", policy=ms.policy)
                return tx, ty, tz, 0.0, 0.0, 0.0, 0.0, "mission_rth_hold"
            return tx, ty, tz, 0.0, 0.0, 0.0, 0.0, "mission_rth"

        elif ms.mode == "traj":
            t_traj = time.time() - ms.hold_start_time
            if ms.traj_type == "fig8":
                A = ms.traj_param
                B = A / 2.0
                omega = 0.05
                tx = home_x + A * math.sin(omega * t_traj)
                ty = home_y + B * math.sin(2.0 * omega * t_traj)
                tz = TARGET_Z
                
                vx = A * omega * math.cos(omega * t_traj)
                vy = B * 2.0 * omega * math.cos(2.0 * omega * t_traj)
                ax = -A * (omega**2) * math.sin(omega * t_traj)
                ay = -B * ((2.0 * omega)**2) * math.sin(2.0 * omega * t_traj)
                
                return tx, ty, tz, vx, vy, ax, ay, "mission_traj_fig8"

            elif ms.traj_type == "circle":
                R = ms.traj_param
                omega = 0.1
                tx = home_x + R * math.cos(omega * t_traj)
                ty = home_y + R * math.sin(omega * t_traj)
                tz = TARGET_Z
                
                vx = -R * omega * math.sin(omega * t_traj)
                vy = R * omega * math.cos(omega * t_traj)
                ax = -R * (omega**2) * math.cos(omega * t_traj)
                ay = -R * (omega**2) * math.sin(omega * t_traj)
                
                return tx, ty, tz, vx, vy, ax, ay, "mission_traj_circle"

    # 沒有任務時待命，不自動起飛
    return current_x, current_y, current_z, 0.0, 0.0, 0.0, 0.0, "idle"

# ==========================================
# 8. 主程式
# ==========================================
VM_IP = "10.203.103.225"
UDP_SEND_PORT = 5005
UDP_RECV_PORT = 5006

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind(("0.0.0.0", UDP_RECV_PORT))
sock_recv.settimeout(0.5)

alt_ctrl = Altitude_ETM_Controller()
etm_pos_x = Fuzzy_ETM_Core(
    OMEGA_POS, F1_POS, F2_POS, AR_POS, "PosX",
    RATE_LIMIT_XY, GAIN_SCALE_XY, POS_DEADZONE_XY, VEL_DEADZONE_XY, SOFT_ERR_XY
)
etm_pos_y = Fuzzy_ETM_Core(
    OMEGA_POS, F1_POS, F2_POS, AR_POS, "PosY",
    RATE_LIMIT_XY, GAIN_SCALE_XY, POS_DEADZONE_XY, VEL_DEADZONE_XY, SOFT_ERR_XY
)

home_initialized = False
home_x, home_y = 0.0, 0.0

log_data = []

UBUNTU_SEND_HZ = 50.0
stat_last_time = time.time()
stat_recv_count = 0
stat_delays = []
current_loss_rate = 0.0
current_avg_delay = 0.0

print("[Hybrid ETM SITL + GAI + Policy] 系統啟動，待命中，等待 Mission CLI 任務...")

cmd_thread = threading.Thread(target=command_thread_fn, daemon=True)
cmd_thread.start()

prev_time = time.time()
last_print_time = 0.0
start_time = time.time()
last_send_time = time.time()

try:
    while True:
        try:
            data, addr = sock_recv.recvfrom(1024)
        except socket.timeout:
            continue

        if len(data) >= 28:
            state_data = struct.unpack("<7f", data[:28])
            x, y, z, vx, vy, vz, yaw = state_data

            echo_ts_windows = struct.unpack("<d", data[28:36])[0] if len(data) >= 36 else 0.0

            if echo_ts_windows > 0.0:
                rtt = time.time() - echo_ts_windows
                delay = rtt / 2.0
                stat_delays.append(abs(delay))
            stat_recv_count += 1

            if not home_initialized:
                home_x = x
                home_y = y
                home_initialized = True
                print(f"[Hybrid ETM SITL + GAI + Policy] Home locked at x={home_x:.2f}, y={home_y:.2f}")

            with MISSION_LOCK:
                mission_runtime_state["x"] = x
                mission_runtime_state["y"] = y
                mission_runtime_state["z"] = z
                mission_runtime_state["yaw"] = yaw
                mission_runtime_state["home_x"] = home_x
                mission_runtime_state["home_y"] = home_y

            curr_time = time.time()
            dt = min(max(curr_time - prev_time, DT_MIN), DT_MAX)
            prev_time = curr_time
            elapsed = curr_time - start_time

            if curr_time - stat_last_time >= 1.0:
                elapsed_stat = curr_time - stat_last_time
                expected_packets = UBUNTU_SEND_HZ * elapsed_stat

                current_loss_rate = max(0.0, 1.0 - (stat_recv_count / expected_packets))
                current_avg_delay = sum(stat_delays) / len(stat_delays) if stat_delays else 0.0

                stat_recv_count = 0
                stat_delays.clear()
                stat_last_time = curr_time

            # 1. 軌跡 / 任務生成
            target_x, target_y, target_z, target_vx, target_vy, target_ax, target_ay, traj_mode = generate_trajectory(
                home_x, home_y, elapsed, x, y, z, yaw
            )

            # 1-1. 依 policy 取得當前控制配置
            with MISSION_LOCK:
                active_policy = mission_state.policy if mission_state.active else "normal"
                mission_mode = mission_state.mode if mission_state.active else "auto_traj"

            policy_cfg = POLICY_CONFIG.get(active_policy, POLICY_CONFIG["normal"])
            xy_scale = policy_cfg["xy_scale"]
            ff_scale = policy_cfg["ff_scale"]
            rp_limit_scale = policy_cfg["rp_limit_scale"]

            # 2. Z 軸控制
            current_alt_state = np.array([z, vz], dtype=float)
            u_accel, trig_z = alt_ctrl.compute_control(current_alt_state, target_z)

            u_accel = float(np.clip(u_accel, -8.0, 8.0))
            thrust_cmd = HOVER_THRUST + (u_accel * THRUST_SCALE)

            if z > ALT_SOFT_CEIL:
                thrust_cmd = min(thrust_cmd, 0.70)
            if z > ALT_HARD_CEIL:
                thrust_cmd = min(thrust_cmd, 0.62)
            thrust_cmd = float(np.clip(thrust_cmd, THR_MIN, THR_MAX))

            # 3. XY 誤差與前饋
            err_w_x = x - target_x
            err_w_y = y - target_y
            err_w_vx = vx - target_vx
            err_w_vy = vy - target_vy

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)

            err_b_x = err_w_x * cos_y + err_w_y * sin_y
            err_b_y = -err_w_x * sin_y + err_w_y * cos_y
            err_b_vx = err_w_vx * cos_y + err_w_vy * sin_y
            err_b_vy = -err_w_vx * sin_y + err_w_vy * cos_y

            ff_b_ax = target_ax * cos_y + target_ay * sin_y
            ff_b_ay = -target_ax * sin_y + target_ay * cos_y

            ff_pitch_cmd = -ff_b_ax * FF_ACCEL_GAIN * ff_scale
            ff_roll_cmd = ff_b_ay * FF_ACCEL_GAIN * ff_scale

            # 模擬陣風
            wind_dist_x = 0.0
            wind_dist_y = 0.0
            wind_flag = ""
            if 20.0 <= elapsed <= 23.0:
                wind_dist_x = 0.0
                wind_dist_y = 0.0
                wind_flag = "⚠️ [陣風干擾中!]"

            wind_b_ax = wind_dist_x * cos_y + wind_dist_y * sin_y
            wind_b_ay = -wind_dist_x * sin_y + wind_dist_y * cos_y
            wind_pitch_effect = -wind_b_ax * FF_ACCEL_GAIN * ff_scale
            wind_roll_effect = wind_b_ay * FF_ACCEL_GAIN * ff_scale

            # 4. XY 控制器更新
            u_pitch, trig_x = etm_pos_x.update(np.array([err_b_x, err_b_vx], dtype=float), 0.0, dt)
            u_roll, trig_y = etm_pos_y.update(np.array([err_b_y, err_b_vy], dtype=float), 0.0, dt)

            u_pitch *= xy_scale
            u_roll *= xy_scale

            policy_rp_limit = ROLL_PITCH_LIMIT * rp_limit_scale

            if z < XY_ENABLE_ALTITUDE:
                low_alt_limit = LOW_ALT_ROLL_PITCH_LIMIT * rp_limit_scale
                target_roll = float(np.clip(-LOW_ALT_XY_GAIN * u_roll, -low_alt_limit, low_alt_limit))
                target_pitch = float(np.clip(-LOW_ALT_XY_GAIN * u_pitch, -low_alt_limit, low_alt_limit))
            else:
                target_roll = float(np.clip(
                    -u_roll + ff_roll_cmd + wind_roll_effect,
                    -policy_rp_limit,
                    policy_rp_limit,
                ))
                target_pitch = float(np.clip(
                    -u_pitch + ff_pitch_cmd + wind_pitch_effect,
                    -policy_rp_limit,
                    policy_rp_limit,
                ))

            # 5. ETM 發送
            time_since_last_send = curr_time - last_send_time
            is_triggered = trig_x or trig_y or trig_z or (time_since_last_send > 0.4)

            if is_triggered:
                windows_now = time.time()
                msg = struct.pack("<4fd", target_roll, target_pitch, 0.0, thrust_cmd, windows_now)
                sock_send.sendto(msg, (VM_IP, UDP_SEND_PORT))
                last_send_time = curr_time

            log_data.append([
                elapsed, x, y, z,
                target_x, target_y, target_z,
                current_loss_rate, current_avg_delay,
                int(is_triggered),
            ])

            # 6. Debug 輸出
            if PRINT_ENABLE and (not CLI_TYPING) and ((curr_time - last_print_time) >= PRINT_PERIOD):
                last_print_time = curr_time
                trigger_str = "TX" if is_triggered else "--"
                print(
                    f"[{traj_mode}] Time:{elapsed:5.1f}s | XYZ=({x:+5.2f},{y:+5.2f},{z:+5.2f}) | "
                    f"Tgt=({target_x:+5.2f},{target_y:+5.2f},{target_z:+5.2f}) | "
                    f"Mission={mission_mode} | Policy={active_policy} | "
                    f"Net: Loss {current_loss_rate*100:4.1f}% Dly {current_avg_delay*1000:4.0f}ms | "
                    f"{trigger_str} {wind_flag}"
                )

except KeyboardInterrupt:
    print("\n[Hybrid ETM SITL + GAI + Policy] 使用者中止控制")
finally:
    sock_send.close()
    sock_recv.close()

# ==========================================
# 9. 存檔與繪圖
# ==========================================
if len(log_data) > 0:
    log_data = np.array(log_data)

    t = log_data[:, 0]
    total_duration = t[-1] - t[0]

    expected_total_packets = total_duration * 50.0
    received_total_packets = len(log_data)

    if expected_total_packets > 0:
        overall_loss_rate = max(0.0, (1.0 - received_total_packets / expected_total_packets)) * 100
    else:
        overall_loss_rate = 0.0

    overall_avg_delay = np.mean(log_data[:, 8]) * 1000

    current_time_str = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"flight_log_{current_time_str}_loss{overall_loss_rate:.1f}pct_delay{overall_avg_delay:.0f}ms.csv"

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Time", "X_real", "Y_real", "Z_real",
            "X_target", "Y_target", "Z_target",
            "Loss_Rate", "Delay_s", "Triggered",
        ])
        writer.writerows(log_data)

    print(f"軌跡與網路數據已儲存至：{csv_filename}")

    x_real, y_real, z_real = log_data[:, 1], log_data[:, 2], log_data[:, 3]
    x_tgt, y_tgt, z_tgt = log_data[:, 4], log_data[:, 5], log_data[:, 6]

    loss_rate_arr = log_data[:, 7] * 100
    delay_ms_arr = log_data[:, 8] * 1000
    triggered_arr = log_data[:, 9]

    e_x = x_real - x_tgt
    e_y = y_real - y_tgt

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(y_tgt, x_tgt, "r--", label="Target Reference")
    plt.plot(y_real, x_real, "b-", label="Actual Flight")
    plt.plot(y_tgt[0], x_tgt[0], "go", label="Start Point")
    plt.xlabel("Y (m)")
    plt.ylabel("X (m)")
    plt.title("Figure-8 / Mission Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, z_tgt, "r--", label="Target Z")
    plt.plot(t, z_real, "b-", label="Actual Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude Z (m)")
    plt.title("Altitude Tracking")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, e_x, "b-", label="Error X", alpha=0.8)
    plt.plot(t, e_y, "g-", label="Error Y", alpha=0.8)
    plt.axvspan(20.0, 23.0, color="red", alpha=0.2, label="Wind Disturbance")
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title("Tracking Error")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t, loss_rate_arr, "r-", label="Packet Loss (%)", alpha=0.5)
    plt.plot(t, delay_ms_arr, "b-", label="Delay (ms)", alpha=0.5)

    trigger_times = t[triggered_arr == 1]
    trigger_vals = np.zeros_like(trigger_times) + 10
    plt.scatter(trigger_times, trigger_vals, color="green", marker="|", alpha=0.5, label="ETM Sent")

    plt.xlabel("Time (s)")
    plt.ylabel("Loss / Delay / Trigger")
    plt.title("Network Statistics & ETM Transmission")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    total_steps = len(log_data)
    triggered_steps = np.sum(triggered_arr)
    saved_steps = total_steps - triggered_steps
    saved_ratio = (saved_steps / total_steps) * 100 if total_steps > 0 else 0

    print("\n=== ETM 頻寬節省與網路統計 ===")
    print(f"整體平均丟包率：{overall_loss_rate:.1f} %")
    print(f"整體平均延遲：  {overall_avg_delay:.0f} ms")
    print(f"總執行迴圈數：  {total_steps} 次")
    print(f"實際發送封包：  {int(triggered_steps)} 次")
    print(f"節省發送封包：  {int(saved_steps)} 次")
    print(f"頻寬節省率：    {saved_ratio:.2f} %")
    print("================================")