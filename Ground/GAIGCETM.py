#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import time
import math
import csv
import json
import numpy as np
import threading
from dataclasses import dataclass, field
from openai import OpenAI

# ==========================================
# 0. OpenAI Client 設定
# ==========================================
client = OpenAI()

# ==========================================
# 1. 控制參數設定
# ==========================================
UAV_MASS = 3.6
HOVER_FORCE = UAV_MASS * 9.81

# 高度控制
OMEGA_ALT = np.array([[11.2683, 16.5707], [16.5707, 25.0074]], dtype=float)
F1_ALT = np.array([-8.1979, -12.7244], dtype=float)
SIGMA_ALT = 0.05
AR_ALT = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)
HOVER_THRUST = 0.52
THRUST_SCALE = 0.045
THR_MIN = 0.10
THR_MAX = 0.90

# XY ETM 控制
OMEGA_POS = np.array([[282.0784, 501.9677], [501.9677, 893.5577]], dtype=float)
F1_POS = np.array([-2.4604, -4.4041], dtype=float)
F2_POS = np.array([-2.9273, -5.2102], dtype=float)
AR_POS = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)

SIGMA_POS = 0.05
POS_ERR_MAX = 1.0
RATE_LIMIT_XY = 3.0
GAIN_SCALE_XY = 0.03
MIN_TRIGGER_INTERVAL = 0.04
TAU_FILTER_XY = 0.15
POS_DEADZONE_XY = 0.001
VEL_DEADZONE_XY = 0.01
SOFT_ERR_XY = 0.01
ALPHA_MIN = 0.15
FF_ACCEL_GAIN = 0.1
ROLL_PITCH_LIMIT = 0.6

DT_MIN = 0.01
DT_MAX = 0.03

# 基本飛行限制
TARGET_Z = 5.0
LANDING_SPEED = 0.5
XY_MAX_RADIUS = 30.0
MIN_Z = 0.3
MAX_Z = 8.0

# 自動腳本與干擾參數
ENABLE_AUTO_SCRIPT = False
AUTO_HOVER_BEFORE_TRAJ = 3.0
FIG8_OMEGA = 0.052
TOTAL_FIG8_TIME = (2.0 * math.pi / FIG8_OMEGA) * 1

DISX, DISY = -6.0, 8.0
DISSTART, DISEND = 30.0, 33.0
PRINT_PERIOD = 0.2

# ==========================================
# 2. Policy 設定
# ==========================================
POLICY_CONFIG = {
    "normal": {"xy_scale": 1.00, "ff_scale": 1.00, "rp_limit_scale": 1.00, "reach_tol": 0.30},
    "smooth": {"xy_scale": 0.75, "ff_scale": 0.70, "rp_limit_scale": 0.75, "reach_tol": 0.35},
    "fast": {"xy_scale": 1.20, "ff_scale": 1.10, "rp_limit_scale": 1.15, "reach_tol": 0.25},
    "safe": {"xy_scale": 0.60, "ff_scale": 0.50, "rp_limit_scale": 0.65, "reach_tol": 0.40},
    "energy_saving": {"xy_scale": 0.65, "ff_scale": 0.55, "rp_limit_scale": 0.70, "reach_tol": 0.45},
}
VALID_POLICIES = set(POLICY_CONFIG.keys())

# ==========================================
# 3. 任務狀態與鎖
# ==========================================
MISSION_LOCK = threading.Lock()

@dataclass
class MissionState:
    mode: str = "idle"       # idle / hover / go_to / relative_move / inspect / run_trajectory / return_home / land / auto
    active: bool = False
    policy: str = "normal"

    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = TARGET_Z

    land_x: float = 0.0
    land_y: float = 0.0

    hold_time: float = 0.0
    hold_start_time: float = 0.0

    traj_type: str = ""
    traj_param: float = 10.0

    confidence: float = 1.0
    ambiguous: bool = False

    plan: list = field(default_factory=list)
    source_text: str = ""

mission_state = MissionState()

runtime_state = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "yaw": 0.0,
    "home_x": 0.0,
    "home_y": 0.0,
}

# ==========================================
# 4. ETM 控制器實作
# ==========================================
class Altitude_ETM_Controller:
    def __init__(self):
        self.last_sent_state = np.zeros(2, dtype=float)
        self.last_control_u = 0.0
        self.ref_state = np.zeros(2, dtype=float)
        self.first_run = True

    def update(self, current_state, target_height, dt):
        r_input = np.array([0.0, 4.0 * target_height], dtype=float)
        self.ref_state += (AR_ALT @ self.ref_state + r_input) * dt

        e_trk = current_state - self.ref_state
        e_net = self.last_sent_state - current_state

        term_net = float(e_net.T @ OMEGA_ALT @ e_net)
        term_trk = float(e_trk.T @ OMEGA_ALT @ e_trk)

        triggered = False
        if self.first_run or (term_net > SIGMA_ALT * term_trk):
            triggered = True
            self.first_run = False
            self.last_sent_state = current_state.copy()
            u = float(np.clip(float(F1_ALT @ e_trk), -8.0, 8.0))
            self.last_control_u = u

        return self.last_control_u, triggered


class Fuzzy_ETM_Core:
    def __init__(self, omega, f1, f2, ar):
        self.Omega = omega
        self.F1 = f1
        self.F2 = f2
        self.Ar = ar

        self.last_sent_error = np.zeros(2, dtype=float)
        self.filtered_error = np.zeros(2, dtype=float)
        self.ref_state = np.zeros(2, dtype=float)
        self.prev_final_u = 0.0
        self.first_run = True
        self.last_trigger_time = 0.0

    def update(self, current_state, target_val, dt, current_time):
        r_input = np.array([0.0, 4.0 * target_val], dtype=float)
        self.ref_state += (self.Ar @ self.ref_state + r_input) * dt

        e_trk = current_state - self.ref_state
        e_net = self.last_sent_error - e_trk

        term_net = float(e_net.T @ self.Omega @ e_net)
        term_trk = float(e_trk.T @ self.Omega @ e_trk)

        triggered = False
        if self.first_run or (
            (term_net > SIGMA_POS * term_trk)
            and (current_time - self.last_trigger_time >= MIN_TRIGGER_INTERVAL)
        ):
            triggered = True
            self.first_run = False
            self.last_trigger_time = current_time
            self.last_sent_error = e_trk.copy()

        self.filtered_error += (dt / TAU_FILTER_XY) * (self.last_sent_error - self.filtered_error)
        e_ctrl = self.filtered_error.copy()

        if abs(e_ctrl[0]) < POS_DEADZONE_XY:
            e_ctrl[0] = 0.0
        if abs(e_ctrl[1]) < VEL_DEADZONE_XY:
            e_ctrl[1] = 0.0

        w2 = float(np.clip(abs(float(e_trk[0])) / POS_ERR_MAX, 0.0, 1.0))
        F_fuzzy = (1.0 - w2) * self.F1 + w2 * self.F2

        e_norm = float(np.linalg.norm(e_ctrl))
        alpha = max(min(1.0, e_norm / SOFT_ERR_XY) if SOFT_ERR_XY > 1e-9 else 1.0, ALPHA_MIN)

        u_raw = float(alpha * (F_fuzzy @ e_ctrl) * GAIN_SCALE_XY)
        du = (u_raw - self.prev_final_u) / dt
        u_final = self.prev_final_u + np.sign(du) * RATE_LIMIT_XY * dt if abs(du) > RATE_LIMIT_XY else u_raw

        self.prev_final_u = float(u_final)
        return self.prev_final_u, triggered

# ==========================================
# 5. 任務工具函式
# ==========================================
def normalize_policy(policy):
    return policy if policy in VALID_POLICIES else "normal"


def clip_target(x, y, z, hx, hy):
    x = float(np.clip(x, hx - XY_MAX_RADIUS, hx + XY_MAX_RADIUS))
    y = float(np.clip(y, hy - XY_MAX_RADIUS, hy + XY_MAX_RADIUS))
    z = float(np.clip(z, MIN_Z, MAX_Z))
    return x, y, z


def body_to_world(dx, dy, yaw):
    return math.cos(yaw) * dx - math.sin(yaw) * dy, math.sin(yaw) * dx + math.cos(yaw) * dy


def clear_mission():
    with MISSION_LOCK:
        mission_state.active = False
        mission_state.mode = "idle"
        mission_state.plan = []
        mission_state.hold_start_time = 0.0
        mission_state.hold_time = 0.0
        mission_state.traj_type = ""
        mission_state.traj_param = 10.0
        mission_state.confidence = 1.0
        mission_state.ambiguous = False


def execute_next_step(curr_x, curr_y, curr_z, curr_yaw, home_x, home_y):
    with MISSION_LOCK:
        if not mission_state.plan:
            mission_state.active = False
            mission_state.mode = "idle"
            return
        step = mission_state.plan.pop(0)

    try:
        apply_parsed_command(step, curr_x, curr_y, curr_z, curr_yaw, home_x, home_y)
    except Exception as e:
        print(f"[Mission] execute_next_step error: {e}")
        clear_mission()


def validate_single_step(cmd):
    if not isinstance(cmd, dict):
        return False, "step is not dict"

    command = cmd.get("command", "")
    if command not in {"hover", "go_to", "relative_move", "inspect", "run_trajectory", "return_home", "land"}:
        return False, "unsupported command"

    confidence = float(cmd.get("confidence", 1.0))
    ambiguous = bool(cmd.get("ambiguous", False))
    if ambiguous:
        return False, "ambiguous step"
    if confidence < 0.7:
        return False, "low confidence"

    policy = normalize_policy(cmd.get("policy", "normal"))
    cmd["policy"] = policy

    frame = cmd.get("frame", "none")
    if command == "go_to" and frame not in {"local", "none"}:
        return False, "go_to must use local frame"

    if command == "relative_move" and frame not in {"body", "none"}:
        return False, "relative_move should use body frame"

    if command in {"go_to", "inspect"}:
        tgt = cmd.get("target", {})
        z = float(tgt.get("z", TARGET_Z))
        if not (MIN_Z <= z <= MAX_Z):
            return False, "target z out of range"

    if command == "run_trajectory":
        traj_type = cmd.get("traj_type", "")
        if traj_type not in {"fig8", "circle"}:
            return False, "unsupported trajectory type"
        traj_param = float(cmd.get("traj_param", 0.0))
        if traj_param <= 0.0 or traj_param > 20.0:
            return False, "trajectory parameter out of range"

    return True, "ok"


def apply_parsed_command(cmd, x, y, z, yaw, hx, hy):
    ok, reason = validate_single_step(cmd)
    if not ok:
        print(f"[GAI] Reject step: {reason}")
        return

    command = cmd.get("command", "hover")
    policy = normalize_policy(cmd.get("policy", "normal"))

    with MISSION_LOCK:
        mission_state.active = True
        mission_state.mode = command
        mission_state.policy = policy
        mission_state.confidence = float(cmd.get("confidence", 1.0))
        mission_state.ambiguous = bool(cmd.get("ambiguous", False))
        mission_state.hold_time = float(cmd.get("hold_time", 0.0))
        mission_state.hold_start_time = 0.0

        if command == "hover":
            mission_state.target_x, mission_state.target_y, mission_state.target_z = x, y, max(z, MIN_Z)

        elif command == "go_to":
            tgt = cmd.get("target", {})
            tx = float(tgt.get("x", x))
            ty = float(tgt.get("y", y))
            tz = float(tgt.get("z", max(z, MIN_Z)))
            mission_state.target_x, mission_state.target_y, mission_state.target_z = clip_target(tx, ty, tz, hx, hy)

        elif command == "relative_move":
            off = cmd.get("offset", {})
            dx = float(off.get("x", 0.0))
            dy = float(off.get("y", 0.0))
            dz = float(off.get("z", 0.0))
            wx, wy = body_to_world(dx, dy, yaw)
            mission_state.target_x, mission_state.target_y, mission_state.target_z = clip_target(
                x + wx, y + wy, z + dz, hx, hy
            )

        elif command == "inspect":
            tgt = cmd.get("target", {})
            tx = float(tgt.get("x", x))
            ty = float(tgt.get("y", y))
            tz = float(tgt.get("z", max(z, MIN_Z)))
            mission_state.target_x, mission_state.target_y, mission_state.target_z = clip_target(tx, ty, tz, hx, hy)
            if mission_state.hold_time <= 0.0:
                mission_state.hold_time = 5.0

        elif command == "run_trajectory":
            mission_state.traj_type = cmd.get("traj_type", "fig8")
            mission_state.traj_param = float(cmd.get("traj_param", 10.0))
            mission_state.target_z = max(z, MIN_Z)
            mission_state.hold_start_time = time.time()

        elif command == "return_home":
            mission_state.target_x, mission_state.target_y, mission_state.target_z = hx, hy, TARGET_Z

        elif command == "land":
            mission_state.land_x, mission_state.land_y = x, y

# ==========================================
# 6. OpenAI 自然語言解析（支援 plan）
# ==========================================
def parse_nl_command_with_openai(user_text, current_state, home_state):
    schema = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["hover", "go_to", "relative_move", "inspect", "run_trajectory", "return_home", "land"]
                        },
                        "frame": {
                            "type": "string",
                            "enum": ["local", "body", "none"]
                        },
                        "target": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"}
                            },
                            "additionalProperties": False
                        },
                        "offset": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"}
                            },
                            "additionalProperties": False
                        },
                        "traj_type": {
                            "type": "string",
                            "enum": ["fig8", "circle", ""]
                        },
                        "traj_param": {"type": "number"},
                        "hold_time": {"type": "number"},
                        "policy": {
                            "type": "string",
                            "enum": ["normal", "smooth", "fast", "safe", "energy_saving"]
                        },
                        "confidence": {"type": "number"},
                        "ambiguous": {"type": "boolean"}
                    },
                    "required": ["command"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["plan"],
        "additionalProperties": False
    }

    prompt = {
        "instruction": "將使用者的無人機中文命令拆解為按順序執行的 plan 陣列，只輸出符合 schema 的 JSON。",
        "rules": [
            "若使用者要求多個動作，請依順序產生多個 step。",
            "go_to 代表絕對座標(local)。",
            "relative_move 代表相對移動(body)。",
            "回原點、回家 對應 return_home。",
            "降落 對應 land。",
            "若語意有歧義，ambiguous=true，confidence 降低。",
            "confidence 範圍 0 到 1。",
            "若未指定 policy，使用 normal。",
            "target / offset 若未使用可以省略，不必強制填滿。",
            f"高度 z 應限制在 {MIN_Z:.2f} 到 {MAX_Z:.2f} 公尺。",
        ],
        "current_state": current_state,
        "home_state": home_state,
        "user_text": user_text,
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "你是無人機高階任務解析器，將中文指令轉為多步驟 JSON plan。"
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False)
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "uav_plan",
                    "schema": schema,
                    "strict": True
                }
            },
            temperature=0.0
        )
        parsed = json.loads(response.choices[0].message.content)

        if "plan" not in parsed or not isinstance(parsed["plan"], list):
            return {"plan": []}

        return parsed

    except Exception as e:
        print(f"[GAI] API 解析失敗: {e}")
        return {"plan": []}

# ==========================================
# 7. 軌跡生成器
# ==========================================
def auto_script_trajectory(hx, hy, elapsed):
    t_hover_end = AUTO_HOVER_BEFORE_TRAJ
    t_fig8_end = t_hover_end + TOTAL_FIG8_TIME

    if elapsed < t_hover_end:
        return hx, hy, TARGET_Z, 0.0, 0.0, 0.0, 0.0, "AutoHover"

    elif elapsed < t_fig8_end:
        t = elapsed - t_hover_end
        A = 10.0
        B = 10.0
        w = FIG8_OMEGA
        tx = hx + A * math.sin(w * t)
        ty = hy + B * math.sin(w * t) * math.cos(w * t)
        tz = TARGET_Z
        tvx = A * w * math.cos(w * t)
        tvy = B * w * math.cos(2.0 * w * t)
        tax = -A * (w ** 2) * math.sin(w * t)
        tay = -2.0 * B * (w ** 2) * math.sin(2.0 * w * t)
        return tx, ty, tz, tvx, tvy, tax, tay, "AutoFig8"

    else:
        tz = max(0.0, TARGET_Z - LANDING_SPEED * (elapsed - t_fig8_end))
        return hx, hy, tz, 0.0, 0.0, 0.0, 0.0, "AutoLand"


def generate_trajectory(hx, hy, elapsed, curr_x, curr_y, curr_z, curr_yaw):
    with MISSION_LOCK:
        ms = MissionState(**mission_state.__dict__)

    if not ms.active:
        if ENABLE_AUTO_SCRIPT:
            return auto_script_trajectory(hx, hy, elapsed)
        return curr_x, curr_y, curr_z, 0.0, 0.0, 0.0, 0.0, "Idle"

    tol = POLICY_CONFIG.get(ms.policy, POLICY_CONFIG["normal"])["reach_tol"]

    if ms.mode in ["go_to", "relative_move", "return_home"]:
        dist = math.sqrt(
            (curr_x - ms.target_x) ** 2 +
            (curr_y - ms.target_y) ** 2 +
            (curr_z - ms.target_z) ** 2
        )
        if dist < tol:
            execute_next_step(curr_x, curr_y, curr_z, curr_yaw, hx, hy)
        return ms.target_x, ms.target_y, ms.target_z, 0.0, 0.0, 0.0, 0.0, ms.mode

    elif ms.mode in ["hover", "inspect"]:
        dist = math.sqrt(
            (curr_x - ms.target_x) ** 2 +
            (curr_y - ms.target_y) ** 2 +
            (curr_z - ms.target_z) ** 2
        )
        if dist < tol:
            if ms.hold_time <= 0.0:
                execute_next_step(curr_x, curr_y, curr_z, curr_yaw, hx, hy)
            else:
                if ms.hold_start_time <= 0.0:
                    with MISSION_LOCK:
                        mission_state.hold_start_time = time.time()
                elif (time.time() - ms.hold_start_time) >= ms.hold_time:
                    execute_next_step(curr_x, curr_y, curr_z, curr_yaw, hx, hy)
        return ms.target_x, ms.target_y, ms.target_z, 0.0, 0.0, 0.0, 0.0, ms.mode

    elif ms.mode == "land":
        if curr_z < 0.05:
            clear_mission()
            return ms.land_x, ms.land_y, 0.0, 0.0, 0.0, 0.0, 0.0, "LandDone"
        tz = max(0.0, curr_z - LANDING_SPEED * DT_MAX)
        return ms.land_x, ms.land_y, tz, 0.0, 0.0, 0.0, 0.0, "Land"

    elif ms.mode == "run_trajectory":
        t = time.time() - ms.hold_start_time
        tz = max(curr_z, MIN_Z)

        if ms.traj_type == "fig8":
            if t > TOTAL_FIG8_TIME:
                execute_next_step(curr_x, curr_y, curr_z, curr_yaw, hx, hy)
                return curr_x, curr_y, curr_z, 0.0, 0.0, 0.0, 0.0, "TrajEnd"

            A = ms.traj_param
            B = ms.traj_param / 2.0
            w = FIG8_OMEGA
            tx = hx + A * math.sin(w * t)
            ty = hy + B * math.sin(2.0 * w * t)
            tvx = A * w * math.cos(w * t)
            tvy = B * 2.0 * w * math.cos(2.0 * w * t)
            tax = -A * (w ** 2) * math.sin(w * t)
            tay = -B * ((2.0 * w) ** 2) * math.sin(2.0 * w * t)
            return tx, ty, tz, tvx, tvy, tax, tay, "Fig8"

        elif ms.traj_type == "circle":
            R = max(0.5, ms.traj_param)
            w = 0.10
            circle_period = 2.0 * math.pi / w

            if t > circle_period:
                execute_next_step(curr_x, curr_y, curr_z, curr_yaw, hx, hy)
                return curr_x, curr_y, curr_z, 0.0, 0.0, 0.0, 0.0, "TrajEnd"

            tx = hx + R * math.cos(w * t)
            ty = hy + R * math.sin(w * t)
            tvx = -R * w * math.sin(w * t)
            tvy = R * w * math.cos(w * t)
            tax = -R * (w ** 2) * math.cos(w * t)
            tay = -R * (w ** 2) * math.sin(w * t)
            return tx, ty, tz, tvx, tvy, tax, tay, "Circle"

    return curr_x, curr_y, curr_z, 0.0, 0.0, 0.0, 0.0, "Idle"

# ==========================================
# 8. CLI 與主迴圈
# ==========================================
def command_thread_fn():
    print("\n[CLI] 輸入 'nl 中文命令'（例：nl 往前飛3公尺然後回起點降落），輸入 q 離開")

    while True:
        cmdline = input("[CLI] > ").strip()
        if not cmdline:
            continue

        if cmdline.lower() in ["q", "quit", "exit"]:
            break

        if cmdline.startswith("nl "):
            with MISSION_LOCK:
                cx = runtime_state["x"]
                cy = runtime_state["y"]
                cz = runtime_state["z"]
                cyaw = runtime_state["yaw"]
                hx = runtime_state["home_x"]
                hy = runtime_state["home_y"]

            c_state = {"x": cx, "y": cy, "z": cz, "yaw": cyaw}
            h_state = {"x": hx, "y": hy, "z": TARGET_Z}

            parsed = parse_nl_command_with_openai(cmdline[3:], c_state, h_state)
            plan_arr = parsed.get("plan", [])

            valid_steps = []
            for i, p in enumerate(plan_arr):
                ok, reason = validate_single_step(p)
                if ok:
                    valid_steps.append(p)
                else:
                    print(f"[GAI] Step {i+1} rejected: {reason}")

            print(f"[GAI] 有效步驟數: {len(valid_steps)}")
            for i, p in enumerate(valid_steps):
                print(f"  Step {i+1}: {p.get('command')}")

            if valid_steps:
                with MISSION_LOCK:
                    mission_state.plan = valid_steps
                execute_next_step(cx, cy, cz, cyaw, hx, hy)

        else:
            print("[CLI] 目前僅支援 'nl' 前綴的自然語言命令。")


def main():
    pi_ip = input("請輸入樹莓派 IP: ").strip()

    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", 5006))
    sock_recv.settimeout(0.5)

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    alt_ctrl = Altitude_ETM_Controller()
    etm_x = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS)
    etm_y = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS)

    log_data = []
    home_initialized = False
    hx = 0.0
    hy = 0.0
    last_roll = 0.0
    last_pitch = 0.0

    start_time = 0.0
    prev_time = 0.0
    last_print_time = 0.0
    last_send_time = 0.0

    print("[GCS] 等待無人機心跳包...")
    threading.Thread(target=command_thread_fn, daemon=True).start()

    try:
        while True:
            try:
                data, _ = sock_recv.recvfrom(1024)
            except socket.timeout:
                continue

            if len(data) != 36:
                continue

            x, y, z, vx, vy, vz, yaw, ts = struct.unpack("<7fd", data)
            curr_time = time.time()

            if not home_initialized:
                hx, hy = x, y
                start_time = curr_time
                prev_time = curr_time
                home_initialized = True

                with MISSION_LOCK:
                    runtime_state["home_x"] = hx
                    runtime_state["home_y"] = hy

                print(f"✅ Home鎖定: ({hx:.2f}, {hy:.2f})")

            dt = min(max(curr_time - prev_time, DT_MIN), DT_MAX)
            prev_time = curr_time
            elapsed = curr_time - start_time

            with MISSION_LOCK:
                runtime_state.update({"x": x, "y": y, "z": z, "yaw": yaw})
                active_policy = mission_state.policy if mission_state.active else "normal"
                mode = mission_state.mode if mission_state.active else "idle"

            cfg = POLICY_CONFIG.get(active_policy, POLICY_CONFIG["normal"])

            # --- 軌跡生成 ---
            tx, ty, tz, tvx, tvy, tax, tay, m_str = generate_trajectory(
                hx, hy, elapsed, x, y, z, yaw
            )

            # --- Z 控制 ---
            u_accel, trig_z = alt_ctrl.update(np.array([z, vz], dtype=float), tz, dt)
            tilt_factor = max(math.cos(last_roll) * math.cos(last_pitch), 0.5)
            thrust = float(np.clip((HOVER_THRUST + u_accel * THRUST_SCALE) / tilt_factor, THR_MIN, THR_MAX))

            # --- 誤差轉 body frame ---
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)

            err_w_x = x - tx
            err_w_y = y - ty
            err_w_vx = vx - tvx
            err_w_vy = vy - tvy

            err_bx = err_w_x * cos_y + err_w_y * sin_y
            err_by = -err_w_x * sin_y + err_w_y * cos_y
            err_bvx = err_w_vx * cos_y + err_w_vy * sin_y
            err_bvy = -err_w_vx * sin_y + err_w_vy * cos_y

            ff_b_ax = tax * cos_y + tay * sin_y
            ff_b_ay = -tax * sin_y + tay * cos_y

            ff_pitch_cmd = -ff_b_ax * FF_ACCEL_GAIN * cfg["ff_scale"]
            ff_roll_cmd = ff_b_ay * FF_ACCEL_GAIN * cfg["ff_scale"]

            # --- 陣風 ---
            wind_b_ax = 0.0
            wind_b_ay = 0.0
            wind_flag = ""

            if DISSTART <= elapsed <= DISEND:
                wind_flag = "⚠️[陣風]"
                wind_b_ax = DISX * cos_y + DISY * sin_y
                wind_b_ay = -DISX * sin_y + DISY * cos_y

            wind_pitch_effect = -wind_b_ax * FF_ACCEL_GAIN * cfg["ff_scale"]
            wind_roll_effect = wind_b_ay * FF_ACCEL_GAIN * cfg["ff_scale"]

            # --- XY ETM ---
            u_pitch, trig_x = etm_x.update(np.array([err_bx, err_bvx]), 0.0, dt, curr_time)
            u_roll, trig_y = etm_y.update(np.array([err_by, err_bvy]), 0.0, dt, curr_time)

            rp_limit = ROLL_PITCH_LIMIT * cfg["rp_limit_scale"]
            last_roll = float(np.clip(-u_roll * cfg["xy_scale"] + ff_roll_cmd + wind_roll_effect, -rp_limit, rp_limit))
            last_pitch = float(np.clip(u_pitch * cfg["xy_scale"] + ff_pitch_cmd + wind_pitch_effect, -rp_limit, rp_limit))

            # --- 傳送 ---
            triggered_now = int(trig_x or trig_y or trig_z)
            if triggered_now or (curr_time - last_send_time > 0.4):
                sock_send.sendto(
                    struct.pack("<4fd", last_roll, last_pitch, 0.0, thrust, curr_time),
                    (pi_ip, 5005)
                )
                last_send_time = curr_time

            # --- 紀錄 ---
            log_data.append([
                elapsed, x, y, z, tx, ty, tz,
                triggered_now, last_roll, last_pitch, thrust,
                active_policy, mode, m_str
            ])

            # --- 顯示 ---
            if curr_time - last_print_time >= PRINT_PERIOD:
                last_print_time = curr_time
                print(
                    f"[{m_str:10}] T:{elapsed:5.1f} | "
                    f"Pos({x:+5.2f},{y:+5.2f},{z:+5.2f}) -> "
                    f"Tgt({tx:+5.2f},{ty:+5.2f},{tz:+5.2f}) | "
                    f"R/P({math.degrees(last_roll):+3.0f},{math.degrees(last_pitch):+3.0f}) "
                    f"| Thr:{thrust:.2f} {wind_flag}"
                )

            # --- 著陸完成 ---
            if mode == "land" and z < 0.2 and abs(vz) < 0.1:
                print("\n[GCS] 著陸確認，鎖定馬達...")
                for _ in range(10):
                    sock_send.sendto(
                        struct.pack("<4fd", 0.0, 0.0, 0.0, 0.0, time.time()),
                        (pi_ip, 5005)
                    )
                    time.sleep(0.05)
                clear_mission()
                break

    except KeyboardInterrupt:
        print("\n[GCS] 中斷。")

    finally:
        sock_recv.close()
        sock_send.close()

        if log_data:
            fname = f"log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time", "X", "Y", "Z",
                    "Tgt_X", "Tgt_Y", "Tgt_Z",
                    "Triggered",
                    "Cmd_Roll", "Cmd_Pitch", "Cmd_Thrust",
                    "Policy", "MissionMode", "TrajectoryMode"
                ])
                writer.writerows(log_data)
            print(f"紀錄已儲存: {fname}")


if __name__ == "__main__":
    main()