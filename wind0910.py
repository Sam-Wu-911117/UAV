import socket
import struct
import time
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

# ==========================================
# 1. 參數設定
# ==========================================

# ---------- Z 軸：舊版高度 ETM ----------
OMEGA_ALT = np.array([
    [11.2683, 16.5707],
    [16.5707, 25.0074]
], dtype=float)

F1_ALT = np.array([-8.1979, -12.7244], dtype=float)
SIGMA_ALT = 0.05

AR_ALT = np.array([
    [0.0, 1.0],
    [-4.0, -4.0]
], dtype=float)

HOVER_THRUST = 0.72
THRUST_SCALE = 0.045

# ---------- XY 軸：ETM (最佳化參數 + 柔化設定) ----------
OMEGA_POS = np.array([
    [282.0784, 501.9677],
    [501.9677, 893.5577]
], dtype=float)

F1_POS = np.array([-2.4604, -4.4041], dtype=float)
F2_POS = np.array([-2.9273, -5.2102], dtype=float)

AR_POS = np.array([
    [0.0, 1.0],
    [-4.0, -4.0]
], dtype=float)

SIGMA_POS = 0.40           # 放寬觸發閾值，消除震盪
POS_ERR_MAX = 1.0

RATE_LIMIT_XY = 3.0
GAIN_SCALE_XY = 0.06       # 大幅降低姿態增益比例，確保平滑
MIN_TRIGGER_INTERVAL = 0.05
TAU_FILTER_XY = 0.15       # 加強速度項低通濾波
POS_DEADZONE_XY = 0.03
VEL_DEADZONE_XY = 0.05
SOFT_ERR_XY = 0.5
ALPHA_MIN = 0.15

# ---------- 前饋控制參數 ----------
FF_ACCEL_GAIN = 0.102 

# ---------- 共用 ----------
ROLL_PITCH_LIMIT = 0.35
DT_MIN = 0.01
DT_MAX = 0.03

TARGET_Z = 5.0
LANDING_SPEED = 0.5      # 降落速度 (m/s)

XY_ENABLE_ALTITUDE = 0.30
LOW_ALT_XY_GAIN = 0.30
LOW_ALT_ROLL_PITCH_LIMIT = 0.03

THR_MIN = 0.30
THR_MAX = 0.90
ALT_SOFT_CEIL = 6.0
ALT_HARD_CEIL = 8.0

# ---------- 8 字軌跡與時序參數 ----------
ENABLE_FIGURE8 = True
HOVER_BEFORE_TRAJ = 8.0  

FIG8_A = 10.0             
FIG8_B = 20.0             
FIG8_OMEGA = 0.05        
FIG8_PERIOD = 2.0 * math.pi / FIG8_OMEGA  # 單趟 8 字所需時間
FIG8_LOOPS = 1                            # 執行趟數
TOTAL_FIG8_TIME = FIG8_PERIOD * FIG8_LOOPS

# ==========================================
# 2. Z 軸控制器
# ==========================================
class Altitude_ETM_Controller:
    def __init__(self):
        self.last_sent_state = np.zeros(2, dtype=float)
        self.last_control_u = 0.0
        self.ref_state = np.zeros(2, dtype=float)
        self.prev_time = time.time()
        self.ref_state[0] = 0.0
        self.first_run = True

    def update_reference(self, dt, target_height):
        r_input = np.array([0.0, 4.0 * target_height], dtype=float)
        dx_r = AR_ALT @ self.ref_state + r_input
        self.ref_state += dx_r * dt
        return self.ref_state

    def compute_control(self, current_state, target_height):
        current_time = time.time()
        dt = current_time - self.prev_time
        dt = float(np.clip(dt, DT_MIN, DT_MAX))
        self.prev_time = current_time

        xr = self.update_reference(dt, target_height)
        e_trk = current_state - xr
        e_net = self.last_sent_state - current_state

        term_net = float(e_net.T @ OMEGA_ALT @ e_net)
        term_trk = float(e_trk.T @ OMEGA_ALT @ e_trk)

        triggered = False
        if self.first_run or (term_net > SIGMA_ALT * term_trk):
            triggered = True
            self.first_run = False
            self.last_sent_state = current_state.copy()
            u = float(F1_ALT @ e_trk)
            self.last_control_u = u
        else:
            u = self.last_control_u

        return u, triggered

# ==========================================
# 3. XY 模糊控制器
# ==========================================
class Fuzzy_ETM_Core:
    def __init__(self, omega, f1, f2, ar, name="Sys",
                 rate_limit=1.5, gain_scale=1.0,
                 pos_deadzone=0.03, vel_deadzone=0.05,
                 soft_err=0.3):
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
        abs_err = abs(float(e_trk[0]))
        w2 = float(np.clip(abs_err / POS_ERR_MAX, 0.0, 1.0))
        w1 = 1.0 - w2
        F_fuzzy = w1 * self.F1 + w2 * self.F2
        return F_fuzzy

    def apply_deadzone(self, e):
        e_out = e.copy()
        if abs(e_out[0]) < self.pos_deadzone:
            e_out[0] = 0.0
        if abs(e_out[1]) < self.vel_deadzone:
            e_out[1] = 0.0
        return e_out

    def soft_scale(self, e):
        e_norm = float(np.linalg.norm(e))
        alpha = min(1.0, e_norm / self.soft_err) if self.soft_err > 1e-9 else 1.0
        alpha = max(alpha, ALPHA_MIN)
        return alpha

    def update(self, current_state, target_val, dt):
        dt = float(np.clip(dt, DT_MIN, DT_MAX))
        now = time.time()

        r_input = np.array([0.0, 4.0 * target_val], dtype=float)
        dx_r = self.Ar @ self.ref_state + r_input
        self.ref_state += dx_r * dt

        e_trk = current_state - self.ref_state
        e_net = self.last_sent_error - e_trk

        term_net = float(e_net.T @ self.Omega @ e_net)
        term_trk = float(e_trk.T @ self.Omega @ e_trk)

        triggered = False
        if self.first_run or (
            (term_net > SIGMA_POS * term_trk) and
            ((now - self.last_trigger_time) >= MIN_TRIGGER_INTERVAL)
        ):
            triggered = True
            self.first_run = False
            self.last_trigger_time = now
            self.last_sent_error = e_trk.copy()

        self.filtered_error += (dt / TAU_FILTER_XY) * (self.last_sent_error - self.filtered_error)
        e_ctrl = self.apply_deadzone(self.filtered_error)

        F_fuzzy = self.get_fuzzy_gain(e_ctrl)
        alpha = self.soft_scale(e_ctrl)

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
# 4. 軌跡狀態機 (懸停 -> 兩趟8字 -> 降落)
# ==========================================
def generate_trajectory(home_x, home_y, elapsed):
    t_hover_end = HOVER_BEFORE_TRAJ
    t_fig8_end = t_hover_end + TOTAL_FIG8_TIME

    if not ENABLE_FIGURE8 or elapsed < t_hover_end:
        # Phase 1: 起飛與懸停
        return home_x, home_y, TARGET_Z, 0.0, 0.0, 0.0, 0.0, "hover"

    elif elapsed < t_fig8_end:
        # Phase 2: 執行 8 字軌跡
        t = elapsed - t_hover_end
        w = FIG8_OMEGA

        target_x = home_x + FIG8_A * math.sin(w * t)
        target_y = home_y + FIG8_B * math.sin(w * t) * math.cos(w * t)
        target_z = TARGET_Z

        target_vx = FIG8_A * w * math.cos(w * t)
        target_vy = FIG8_B * w * math.cos(2.0 * w * t)
        target_ax = -FIG8_A * (w**2) * math.sin(w * t)
        target_ay = -2.0 * FIG8_B * (w**2) * math.sin(2.0 * w * t)

        return target_x, target_y, target_z, target_vx, target_vy, target_ax, target_ay, "fig8"

    else:
        # Phase 3: 降落
        t_land = elapsed - t_fig8_end
        target_x = home_x
        target_y = home_y
        # 高度線性遞減，直到 0
        target_z = max(0.0, TARGET_Z - LANDING_SPEED * t_land)
        
        return target_x, target_y, target_z, 0.0, 0.0, 0.0, 0.0, "land"

# ==========================================
# 5. 主程式
# ==========================================
VM_IP = "10.159.37.86"
UDP_SEND_PORT = 5005
UDP_RECV_PORT = 5006

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind(("0.0.0.0", UDP_RECV_PORT))
sock_recv.settimeout(0.5)

alt_ctrl = Altitude_ETM_Controller()
etm_pos_x = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS, "PosX", RATE_LIMIT_XY, GAIN_SCALE_XY, POS_DEADZONE_XY, VEL_DEADZONE_XY, SOFT_ERR_XY)
etm_pos_y = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS, "PosY", RATE_LIMIT_XY, GAIN_SCALE_XY, POS_DEADZONE_XY, VEL_DEADZONE_XY, SOFT_ERR_XY)

home_initialized = False
home_x, home_y = 0.0, 0.0

# 紀錄飛行數據用的 List
log_data = []

print("[Hybrid ETM SITL] 系統啟動，準備執行起飛 -> 8 字繞行 (含陣風測試) -> 降落")
prev_time = time.time()
last_print_time = 0.0
start_time = time.time()

try:
    while True:
        try:
            data, addr = sock_recv.recvfrom(1024)
        except socket.timeout:
            continue

        if len(data) >= 28:
            state_data = struct.unpack('<7f', data[:28])
            x, y, z, vx, vy, vz, yaw = state_data
            ts_ubuntu = struct.unpack('<d', data[28:36])[0] if len(data) >= 36 else 0.0

            if not home_initialized:
                home_x = x
                home_y = y
                home_initialized = True
                print(f"[Hybrid ETM SITL] Home locked at x={home_x:.2f}, y={home_y:.2f}")

            curr_time = time.time()
            dt = min(max(curr_time - prev_time, DT_MIN), DT_MAX)
            prev_time = curr_time
            elapsed = curr_time - start_time

            # 1. 軌跡生成
            target_x, target_y, target_z, target_vx, target_vy, target_ax, target_ay, traj_mode = generate_trajectory(home_x, home_y, elapsed)

            # 2. Z 軸控制
            current_alt_state = np.array([z, vz], dtype=float)
            u_accel, _ = alt_ctrl.compute_control(current_alt_state, target_z)

            u_accel = float(np.clip(u_accel, -8.0, 8.0))
            thrust_cmd = HOVER_THRUST + (u_accel * THRUST_SCALE)

            if z > ALT_SOFT_CEIL: thrust_cmd = min(thrust_cmd, 0.70)
            if z > ALT_HARD_CEIL: thrust_cmd = min(thrust_cmd, 0.62)
            thrust_cmd = float(np.clip(thrust_cmd, THR_MIN, THR_MAX))

            # ==========================================
            # 3. XY 軸誤差與前饋計算
            # ==========================================
            err_w_x = x - target_x
            err_w_y = y - target_y
            err_w_vx = vx - target_vx
            err_w_vy = vy - target_vy

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)

            # 世界座標誤差 -> 機身座標誤差
            err_b_x = err_w_x * cos_y + err_w_y * sin_y
            err_b_y = -err_w_x * sin_y + err_w_y * cos_y
            err_b_vx = err_w_vx * cos_y + err_w_vy * sin_y
            err_b_vy = -err_w_vx * sin_y + err_w_vy * cos_y

            # 世界座標目標加速度 -> 機身座標前饋加速度
            ff_b_ax = target_ax * cos_y + target_ay * sin_y
            ff_b_ay = -target_ax * sin_y + target_ay * cos_y

            ff_pitch_cmd = -ff_b_ax * FF_ACCEL_GAIN
            ff_roll_cmd = ff_b_ay * FF_ACCEL_GAIN

            # ==========================================
            # [新增] 模擬限時陣風干擾 (Wind Gust Injection)
            # ==========================================
            wind_dist_x = 0.0
            wind_dist_y = 0.0
            wind_flag = ""

            # 設定在第 20 秒到 23 秒之間，遭受強烈的側向陣風
            if 20.0 <= elapsed <= 23.0:
                wind_dist_x = 2.0  # 模擬 X 方向的持續強風等效加速度 (m/s^2)
                wind_dist_y = 2.0  # 模擬 Y 方向的持續強風等效加速度 (m/s^2)
                wind_flag = "⚠️ [陣風干擾中!]"

            # 將風擾動等效轉換為機身座標系 (Body Frame)
            wind_b_ax = wind_dist_x * cos_y + wind_dist_y * sin_y
            wind_b_ay = -wind_dist_x * sin_y + wind_dist_y * cos_y
            
            # 將風干擾轉化為對姿態的物理衝擊
            wind_pitch_effect = -wind_b_ax * FF_ACCEL_GAIN
            wind_roll_effect  = wind_b_ay * FF_ACCEL_GAIN

            # ==========================================
            # 4. XY 控制器更新
            # ==========================================
            u_pitch, trig_x = etm_pos_x.update(np.array([err_b_x, err_b_vx], dtype=float), 0.0, dt)
            u_roll, trig_y = etm_pos_y.update(np.array([err_b_y, err_b_vy], dtype=float), 0.0, dt)

            # 合併回饋(Feedback)、前饋(Feedforward) 與 陣風干擾(Wind Disturbance)
            if z < XY_ENABLE_ALTITUDE:
                target_roll = float(np.clip(-LOW_ALT_XY_GAIN * u_roll, -LOW_ALT_ROLL_PITCH_LIMIT, LOW_ALT_ROLL_PITCH_LIMIT))
                target_pitch = float(np.clip(-LOW_ALT_XY_GAIN * u_pitch, -LOW_ALT_ROLL_PITCH_LIMIT, LOW_ALT_ROLL_PITCH_LIMIT))
            else:
                # 把 wind_roll_effect 和 wind_pitch_effect 強加進去，當作未知的外部擾動
                target_roll = float(np.clip(-u_roll + ff_roll_cmd + wind_roll_effect, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                target_pitch = float(np.clip(-u_pitch + ff_pitch_cmd + wind_pitch_effect, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))

            # ==========================================
            # 5. 發送指令與紀錄資料
            # ==========================================
            msg = struct.pack('<4fd', target_roll, target_pitch, 0.0, thrust_cmd, ts_ubuntu)
            sock_send.sendto(msg, (VM_IP, UDP_SEND_PORT))

            log_data.append([elapsed, x, y, z, target_x, target_y, target_z])

            # ==========================================
            # 6. Debug 輸出與降落終止條件
            # ==========================================
            if (curr_time - last_print_time) >= 0.2:
                last_print_time = curr_time
                print(f"[{traj_mode}] Time:{elapsed:5.1f}s | XYZ=({x:+5.2f},{y:+5.2f},{z:+5.2f}) | Tgt=({target_x:+.2f},{target_y:+.2f},{target_z:+.2f}) {wind_flag}")

            # 當模式為降落，且目標高度歸零、實際高度極低時，終止迴圈
            if traj_mode == "land" and target_z <= 0.0 and z < 0.2:
                # 送出關停馬達指令
                stop_msg = struct.pack('<4fd', 0.0, 0.0, 0.0, 0.0, ts_ubuntu)
                for _ in range(5):
                    sock_send.sendto(stop_msg, (VM_IP, UDP_SEND_PORT))
                    time.sleep(0.05)
                print("\n[Hybrid ETM SITL] 降落完成，馬達關閉。準備匯出紀錄...")
                break

except KeyboardInterrupt:
    print("\n[Hybrid ETM SITL] 使用者中止控制")
finally:
    sock_send.close()
    sock_recv.close()

# ==========================================
# 6. 存檔與繪圖
# ==========================================
if len(log_data) > 0:
    log_data = np.array(log_data)
    
    # 存成 CSV 檔案
    with open('flight_log0310.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'X_real', 'Y_real', 'Z_real', 'X_target', 'Y_target', 'Z_target'])
        writer.writerows(log_data)
    print("軌跡數據已儲存至 flight_log0310.csv")

    # 繪圖
    t = log_data[:, 0]
    x_real, y_real, z_real = log_data[:, 1], log_data[:, 2], log_data[:, 3]
    x_tgt, y_tgt, z_tgt = log_data[:, 4], log_data[:, 5], log_data[:, 6]
    
    e_x = x_real - x_tgt
    e_y = y_real - y_tgt

    plt.figure(figsize=(15, 5))

    # 子圖 1: XY 2D 軌跡圖
    plt.subplot(1, 3, 1)
    plt.plot(y_tgt, x_tgt, 'r--', label='Target Reference')
    plt.plot(y_real, x_real, 'b-', label='Actual Flight')
    plt.plot(y_tgt[0], x_tgt[0], 'go', label='Start Point')
    plt.xlabel('Y (m)')
    plt.ylabel('X (m)')
    plt.title('Figure-8 Trajectory')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # 子圖 2: 高度追蹤圖
    plt.subplot(1, 3, 2)
    plt.plot(t, z_tgt, 'r--', label='Target Z')
    plt.plot(t, z_real, 'b-', label='Actual Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude Z (m)')
    plt.title('Altitude Tracking')
    plt.grid(True)
    plt.legend()

    # 子圖 3: XY 誤差與陣風區間 (論文專用)
    plt.subplot(1, 3, 3)
    plt.plot(t, e_x, 'b-', label='Error X', alpha=0.8)
    plt.plot(t, e_y, 'g-', label='Error Y', alpha=0.8)
    
    # 標示陣風干擾區間 (t=20 ~ t=23)
    plt.axvspan(20.0, 23.0, color='red', alpha=0.2, label='Wind Disturbance')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (m)')
    plt.title('Tracking Error (Step Response)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()