#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import time
import math
import csv
import numpy as np
from pathlib import Path

# ==========================================
# 1. 控制參數設定 (對齊 init_params.m)
# ==========================================
# UAV 參數
UAV_MASS = 3.6
HOVER_FORCE = UAV_MASS * 9.81

# 高度控制
OMEGA_ALT = np.array([[11.2683, 16.5707], [16.5707, 25.0074]], dtype=float)
F1_ALT = np.array([-8.1979, -12.7244], dtype=float)
SIGMA_ALT = 0.00
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

SIGMA_POS = 0.00
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

# 軌跡與任務參數
TARGET_Z = 5.0
LANDING_SPEED = 0.5
ENABLE_FIGURE8 = True
HOVER_BEFORE_TRAJ = 3.0
FIG8_A = 10.0
FIG8_B = 10.0
FIG8_OMEGA = 0.052
FIG8_PERIOD = 2.0 * math.pi / FIG8_OMEGA
FIG8_LOOPS = 1
TOTAL_FIG8_TIME = FIG8_PERIOD * FIG8_LOOPS

# 陣風模擬參數
DISX = -6.0
DISY = 8.0
DISSTART = 30.0
DISEND = 33.0

# ==========================================
# 2. ETM 核心控制器類別
# ==========================================
class Altitude_ETM_Controller:
    def __init__(self):
        self.last_sent_state = np.zeros(2, dtype=float)
        self.last_control_u = 0.0
        self.ref_state = np.zeros(2, dtype=float)
        self.first_run = True

    def update(self, current_state, target_height, dt):
        # 參考模型更新
        r_input = np.array([0.0, 4.0 * target_height], dtype=float)
        dx_r = AR_ALT @ self.ref_state + r_input
        self.ref_state += dx_r * dt

        e_trk = current_state - self.ref_state
        e_net = self.last_sent_state - current_state

        term_net = float(e_net.T @ OMEGA_ALT @ e_net)
        term_trk = float(e_trk.T @ OMEGA_ALT @ e_trk)

        triggered = False
        if self.first_run or (term_net > SIGMA_ALT * term_trk):
            triggered = True
            self.first_run = False
            self.last_sent_state = current_state.copy()
            u = float(F1_ALT @ e_trk)
            u = float(np.clip(u, -8.0, 8.0))
            self.last_control_u = u
        else:
            u = self.last_control_u

        return u, triggered

class Fuzzy_ETM_Core:
    def __init__(self, omega, f1, f2, ar):
        self.Omega = omega
        self.F1 = f1
        self.F2 = f2
        self.Ar = ar

        self.last_sent_error = np.zeros(2, dtype=float)
        self.filtered_error = np.zeros(2, dtype=float)
        self.last_control_u = 0.0
        self.ref_state = np.zeros(2, dtype=float)
        self.prev_final_u = 0.0
        self.first_run = True
        self.last_trigger_time = 0.0

    def update(self, current_state, target_val, dt, current_time):
        # 參考模型更新
        r_input = np.array([0.0, 4.0 * target_val], dtype=float)
        dx_r = self.Ar @ self.ref_state + r_input
        self.ref_state += dx_r * dt

        e_trk = current_state - self.ref_state
        e_net = self.last_sent_error - e_trk

        term_net = float(e_net.T @ self.Omega @ e_net)
        term_trk = float(e_trk.T @ self.Omega @ e_trk)

        triggered = False
        if self.first_run or ((term_net > SIGMA_POS * term_trk) and 
                              ((current_time - self.last_trigger_time) >= MIN_TRIGGER_INTERVAL)):
            triggered = True
            self.first_run = False
            self.last_trigger_time = current_time
            self.last_sent_error = e_trk.copy()

        # 低通濾波
        self.filtered_error += (dt / TAU_FILTER_XY) * (self.last_sent_error - self.filtered_error)
        e_ctrl = self.filtered_error.copy()
        
        # 誤差死區限制
        if abs(e_ctrl[0]) < POS_DEADZONE_XY: e_ctrl[0] = 0.0
        if abs(e_ctrl[1]) < VEL_DEADZONE_XY: e_ctrl[1] = 0.0

        # Fuzzy 權重計算
        abs_err = abs(float(e_trk[0]))
        w2 = float(np.clip(abs_err / POS_ERR_MAX, 0.0, 1.0))
        w1 = 1.0 - w2
        F_fuzzy = w1 * self.F1 + w2 * self.F2
        
        # Soft Scale (Alpha)
        e_norm = float(np.linalg.norm(e_ctrl))
        alpha = min(1.0, e_norm / SOFT_ERR_XY) if SOFT_ERR_XY > 1e-9 else 1.0
        alpha = max(alpha, ALPHA_MIN)

        # 基礎控制律計算與速率限制
        u_raw = float(alpha * (F_fuzzy @ e_ctrl) * GAIN_SCALE_XY)
        du = (u_raw - self.prev_final_u) / dt
        if abs(du) > RATE_LIMIT_XY:
            u_final = self.prev_final_u + np.sign(du) * RATE_LIMIT_XY * dt
        else:
            u_final = u_raw

        self.last_control_u = float(u_final)
        self.prev_final_u = self.last_control_u

        return self.last_control_u, triggered

# ==========================================
# 3. 軌跡生成器
# ==========================================
def generate_trajectory(home_x, home_y, elapsed):
    t_hover_end = HOVER_BEFORE_TRAJ
    t_fig8_end = t_hover_end + TOTAL_FIG8_TIME

    if not ENABLE_FIGURE8 or elapsed < t_hover_end:
        return home_x, home_y, TARGET_Z, 0.0, 0.0, 0.0, 0.0, "Hover"
    elif elapsed < t_fig8_end:
        t = elapsed - t_hover_end
        w = FIG8_OMEGA
        target_x = home_x + FIG8_A * math.sin(w * t)
        target_y = home_y + FIG8_B * math.sin(w * t) * math.cos(w * t)
        target_z = TARGET_Z
        target_vx = FIG8_A * w * math.cos(w * t)
        target_vy = FIG8_B * w * math.cos(2.0 * w * t)
        target_ax = -FIG8_A * (w**2) * math.sin(w * t)
        target_ay = -2.0 * FIG8_B * (w**2) * math.sin(2.0 * w * t)
        return target_x, target_y, target_z, target_vx, target_vy, target_ax, target_ay, "Fig8"
    else:
        t_land = elapsed - t_fig8_end
        target_z = max(0.0, TARGET_Z - LANDING_SPEED * t_land)
        return home_x, home_y, target_z, 0.0, 0.0, 0.0, 0.0, "Land"

# ==========================================
# 4. 主程式
# ==========================================
def main():
    PI_IP = input("請輸入樹莓派 IP: ").strip()
    UDP_LISTEN_PORT = 5006  # 接收 Pi 的狀態
    UDP_SEND_PORT = 5005    # 發送指令給 Pi

    # 初始化通訊
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", UDP_LISTEN_PORT))
    sock_recv.settimeout(0.5)
    
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 初始化控制器
    alt_ctrl = Altitude_ETM_Controller()
    etm_pos_x = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS)
    etm_pos_y = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS)

    # 紀錄與狀態變數
    log_data = []
    home_initialized = False
    home_x = home_y = 0.0
    
    # 姿態記憶 (用於推力傾角補償)
    last_cmd_roll = 0.0
    last_cmd_pitch = 0.0

    print(f"\n[Ground Controller] 監聽 Port {UDP_LISTEN_PORT}，準備發送至 {PI_IP}:{UDP_SEND_PORT}")
    print("[Ground Controller] 等待無人機心跳包...")

    start_time = None
    prev_time = None
    last_print_time = 0.0
    last_send_time = 0.0

    try:
        while True:
            try:
                data, addr = sock_recv.recvfrom(1024)
            except socket.timeout:
                continue

            # 接收 Pi 的狀態 <7fd> = 7*4 + 8 = 36 bytes
            if len(data) == 36:
                x, y, z, vx, vy, vz, yaw, pi_ts = struct.unpack("<7fd", data)
                curr_time = time.time()

                if not home_initialized:
                    home_x, home_y = x, y
                    start_time = curr_time
                    prev_time = curr_time
                    home_initialized = True
                    print(f"✅ Home 鎖定完畢: X={home_x:.2f}, Y={home_y:.2f}。控制迴圈啟動！")

                dt = min(max(curr_time - prev_time, DT_MIN), DT_MAX)
                prev_time = curr_time
                elapsed = curr_time - start_time

                # 1. 軌跡生成
                tgt_x, tgt_y, tgt_z, tgt_vx, tgt_vy, tgt_ax, tgt_ay, mode = generate_trajectory(home_x, home_y, elapsed)

                # 2. Z 軸 ETM 與推力傾角補償
                u_accel, trig_z = alt_ctrl.update(np.array([z, vz], dtype=float), tgt_z, dt)
                
                thrust_z_norm = HOVER_THRUST + (u_accel * THRUST_SCALE)
                # 使用前一次的命令角度進行補償
                tilt_factor = math.cos(last_cmd_roll) * math.cos(last_cmd_pitch)
                tilt_factor = max(tilt_factor, 0.5)
                
                thrust_cmd_norm = thrust_z_norm / tilt_factor
                thrust_cmd = float(np.clip(thrust_cmd_norm, THR_MIN, THR_MAX))

                # 3. XY 軸誤差與前饋計算
                err_w_x = x - tgt_x
                err_w_y = y - tgt_y
                err_w_vx = vx - tgt_vx
                err_w_vy = vy - tgt_vy

                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)

                err_b_x = err_w_x * cos_y + err_w_y * sin_y
                err_b_y = -err_w_x * sin_y + err_w_y * cos_y
                err_b_vx = err_w_vx * cos_y + err_w_vy * sin_y
                err_b_vy = -err_w_vx * sin_y + err_w_vy * cos_y

                ff_b_ax = tgt_ax * cos_y + tgt_ay * sin_y
                ff_b_ay = -tgt_ax * sin_y + tgt_ay * cos_y
                
                ff_pitch_cmd = -ff_b_ax * FF_ACCEL_GAIN
                ff_roll_cmd = ff_b_ay * FF_ACCEL_GAIN

                # 4. 陣風干擾前饋 (從 30s 到 33s)
                wind_b_ax = wind_b_ay = 0.0
                wind_flag = ""
                if DISSTART <= elapsed <= DISEND:
                    wind_flag = "⚠️[陣風]"
                    wind_b_ax = DISX * cos_y + DISY * sin_y
                    wind_b_ay = -DISX * sin_y + DISY * cos_y
                
                wind_pitch_effect = -wind_b_ax * FF_ACCEL_GAIN
                wind_roll_effect  = wind_b_ay * FF_ACCEL_GAIN

                # 5. XY ETM 更新
                u_pitch, trig_x = etm_pos_x.update(np.array([err_b_x, err_b_vx], dtype=float), 0.0, dt, curr_time)
                u_roll, trig_y = etm_pos_y.update(np.array([err_b_y, err_b_vy], dtype=float), 0.0, dt, curr_time)

                target_roll = float(np.clip(-u_roll + ff_roll_cmd + wind_roll_effect, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                target_pitch = float(np.clip(u_pitch + ff_pitch_cmd + wind_pitch_effect, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                
                # 記憶角度以供下一次迴圈進行傾角補償
                last_cmd_roll = target_roll
                last_cmd_pitch = target_pitch

                # 6. ETM 觸發判斷與封包發送
                is_triggered = int(trig_x or trig_y or trig_z)
                time_since_last_send = curr_time - last_send_time
                
                # 若 ETM 觸發，或超過 0.4s (防止 Pi 端的 CMD_TIMEOUT=0.5 觸發安全模式)，則發送
                if is_triggered or (time_since_last_send > 0.4):
                    # 打包給 Pi: <4fd> = roll, pitch, yaw_rate, thrust, pc_ts
                    msg = struct.pack("<4fd", target_roll, target_pitch, 0.0, thrust_cmd, curr_time)
                    sock_send.sendto(msg, (PI_IP, UDP_SEND_PORT))
                    last_send_time = curr_time

                # 紀錄資料供繪圖
                log_data.append([elapsed, x, y, z, tgt_x, tgt_y, tgt_z, is_triggered])

                # 終端機狀態輸出
                if curr_time - last_print_time >= 0.2:
                    last_print_time = curr_time
                    trig_str = "TX" if is_triggered else "--"
                    print(f"[{mode:5}] T:{elapsed:5.1f}s | XYZ({x:+4.1f},{y:+4.1f},{z:+4.1f}) | "
                          f"CMD R/P({math.degrees(target_roll):+4.0f}°,{math.degrees(target_pitch):+4.0f}°) Thr:{thrust_cmd:.2f} | {trig_str} {wind_flag}")

                # 降落終止條件
                if mode == "Land" and tgt_z <= 0.0 and z < 0.2:
                    print("\n[Ground Controller] 降落完成，發送安全鎖定指令...")
                    stop_msg = struct.pack("<4fd", 0.0, 0.0, 0.0, 0.0, time.time())
                    for _ in range(10):
                        sock_send.sendto(stop_msg, (PI_IP, UDP_SEND_PORT))
                        time.sleep(0.05)
                    break

    except KeyboardInterrupt:
        print("\n[Ground Controller] 使用者強制中斷控制")
    finally:
        sock_recv.close()
        sock_send.close()
        
        # 匯出 CSV 檔案
        if log_data:
            filename = f"flight_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z', 'Tgt_X', 'Tgt_Y', 'Tgt_Z', 'Triggered'])
                writer.writerows(log_data)
            print(f"[Ground Controller] 飛行紀錄已儲存至 {filename}")

if __name__ == "__main__":
    main()