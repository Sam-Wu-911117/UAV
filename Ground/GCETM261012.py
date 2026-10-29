#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import time
import math
import csv
import json
import os
import numpy as np

# ==========================================
# 1. 控制參數設定 (保留新區間二型模糊強健控制增益)
# ==========================================
UAV_MASS = 3.5
HOVER_FORCE = UAV_MASS * 9.81

# 高度控制矩陣
OMEGA_ALT = np.array([[11.2683, 16.5707], [16.5707, 25.0074]], dtype=float)
F1_ALT = np.array([-8.1979, -12.7244], dtype=float)
SIGMA_ALT = 0.00
AR_ALT = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)
HOVER_THRUST = 0.65
THRUST_SCALE = 0.045
THR_MIN, THR_MAX = 0.10, 0.90

# 水平姿態控制矩陣 (X/Y共用)
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
DT_MIN, DT_MAX = 0.01, 0.03

# ==========================================
# 2. VLA 視覺伺服與泛化巡檢參數 (無固定航線、無雲台)
# ==========================================
TARGET_Z = 2.0                 # 初始起飛與搜索基準高度
HOVER_BEFORE_TRAJ = 3.0        # 起飛後穩定懸停時間
TRANSITION_SPEED = 0.4         # 跨樹轉移時的側向平移速度 (m/s)
SEARCH_YAW_RATE = 0.15         # 原地搜索目標時的偏航角速度 (rad/s)
LANDING_SPEED = 0.5            # 降落速度

# 2D 像素誤差轉換增益 (Pixel Error to Body Frame Error)
K_VLA_LATERAL = 0.002          # 像素 X 誤差 -> 機身 Y 軸誤差 (控制 Roll)
K_VLA_FORWARD = 0.05           # 面積比例誤差 -> 機身 X 軸誤差 (控制 Pitch 距離)
K_VLA_ALTITUDE = 0.001         # 像素 Y 誤差 -> 高度 Z 軸目標微調
K_VLA_YAW = 0.001              # 像素 X 誤差 -> 偏航角微調 (將目標對齊中心)

SHM_VLA_ACT_PATH = '/dev/shm/vla_feedback.json'

# ==========================================
# 3. VLA 決策讀取函式
# ==========================================
def read_vla_feedback():
    """
    讀取 VLA 輸出的即時視覺特徵與語意狀態
    預期 JSON 格式:
    {
      "semantic_state": "TRACKING", // 狀態: SEARCH, TRACKING, TRANSITION, DONE
      "pixel_err_x": 120,           // 影像左右像素誤差 (目標在右為正)
      "pixel_err_y": -30,           // 影像上下像素誤差 (目標在下為負)
      "area_err": 0.5,              // 目標大小誤差 (面積太小為正 -> 需前進)
      "confidence": 0.95
    }
    """
    if not os.path.exists(SHM_VLA_ACT_PATH):
        return None
    try:
        with open(SHM_VLA_ACT_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return None

# ==========================================
# 4. 新區間二型分段多項式模糊網路控制器 (ETM)
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
        self.Omega, self.F1, self.F2, self.Ar = omega, f1, f2, ar
        self.last_sent_error, self.filtered_error = np.zeros(2), np.zeros(2)
        self.ref_state = np.zeros(2, dtype=float)
        self.last_control_u, self.prev_final_u = 0.0, 0.0
        self.first_run, self.last_trigger_time = True, 0.0

    def update(self, error_body_frame, error_vel_body, dt, current_time):
        """
        將 VLA 降維映射後的機身體系誤差直接輸入控制器，
        透過歸屬函數處理非線性擾動與視覺延遲。
        """
        e_trk = np.array([error_body_frame, error_vel_body], dtype=float)
        e_net = self.last_sent_error - e_trk
        
        term_net = float(e_net.T @ self.Omega @ e_net)
        term_trk = float(e_trk.T @ self.Omega @ e_trk)
        
        triggered = False
        if self.first_run or ((term_net > SIGMA_POS * term_trk) and ((current_time - self.last_trigger_time) >= MIN_TRIGGER_INTERVAL)):
            triggered = True
            self.first_run = False
            self.last_trigger_time = current_time
            self.last_sent_error = e_trk.copy()
            
        self.filtered_error += (dt / TAU_FILTER_XY) * (self.last_sent_error - self.filtered_error)
        e_ctrl = self.filtered_error.copy()
        
        if abs(e_ctrl[0]) < POS_DEADZONE_XY: e_ctrl[0] = 0.0
        if abs(e_ctrl[1]) < VEL_DEADZONE_XY: e_ctrl[1] = 0.0
        
        # 核心：依據追蹤誤差計算區間二型模糊歸屬函數 (Membership Function) 權重 w2
        w2 = float(np.clip(abs(float(e_trk[0])) / POS_ERR_MAX, 0.0, 1.0))
        F_fuzzy = (1.0 - w2) * self.F1 + w2 * self.F2
        
        alpha = max(min(1.0, float(np.linalg.norm(e_ctrl)) / SOFT_ERR_XY), ALPHA_MIN)
        u_raw = float(alpha * (F_fuzzy @ e_ctrl) * GAIN_SCALE_XY)
        du = (u_raw - self.prev_final_u) / dt
        u_final = self.prev_final_u + np.sign(du) * RATE_LIMIT_XY * dt if abs(du) > RATE_LIMIT_XY else u_raw
        
        self.last_control_u = self.prev_final_u = float(u_final)
        return self.last_control_u, triggered

# ==========================================
# 5. 主程式
# ==========================================
def main():
    if os.path.exists(SHM_VLA_ACT_PATH):
        try: os.remove(SHM_VLA_ACT_PATH)
        except OSError: pass

    PI_IP = input("請輸入樹莓派 IP (本地模擬請輸入 127.0.0.1): ").strip()
    UDP_LISTEN_PORT, UDP_SEND_PORT = 5006, 5005

    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", UDP_LISTEN_PORT))
    sock_recv.settimeout(0.5)
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    alt_ctrl = Altitude_ETM_Controller()
    etm_pos_x = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS) # 控制機身 X (Pitch)
    etm_pos_y = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS) # 控制機身 Y (Roll)
    
    log_data = []
    home_initialized = False
    home_x = home_y = 0.0
    last_cmd_roll = last_cmd_pitch = target_yaw = 0.0

    flight_state = "HOVER_HOME"
    target_z = TARGET_Z
    transition_start_time = 0.0
    tree_count = 0

    print("\n[Ground Controller] 已啟用【無雲台 VLA 泛化追蹤】模式！")
    print("[Ground Controller] 等待心跳包...")

    start_time = prev_time = None
    last_print_time = last_send_time = 0.0

    try:
        while True:
            try:
                data, addr = sock_recv.recvfrom(1024)
            except socket.timeout:
                continue

            if len(data) == 36:
                x, y, z, vx, vy, vz, yaw, pi_ts = struct.unpack("<7fd", data)
                curr_time = time.time()

                if not home_initialized:
                    home_x, home_y = x, y
                    target_yaw = yaw
                    start_time = prev_time = curr_time
                    home_initialized = True
                    print(f"✅ Home 鎖定: X={home_x:.2f}, Y={home_y:.2f}。準備爬升至 {TARGET_Z}m...")

                dt = min(max(curr_time - prev_time, DT_MIN), DT_MAX)
                prev_time = curr_time
                elapsed = curr_time - start_time

                # 直接生成機身座標系的控制誤差 (Body Frame Errors)
                err_b_x = 0.0
                err_b_y = 0.0
                err_b_vx = vx * math.cos(yaw) + vy * math.sin(yaw)  # 估測速度投影至機身 X
                err_b_vy = -vx * math.sin(yaw) + vy * math.cos(yaw) # 估測速度投影至機身 Y
                
                # 讀取高階 VLA 語意指令
                vla_data = read_vla_feedback()
                vla_state = vla_data.get("semantic_state", "SEARCH") if vla_data else "SEARCH"

                # --------------------------------------------------
                # 狀態機與 VLA 控制映射
                # --------------------------------------------------
                if flight_state == "HOVER_HOME":
                    mode = "Hover_Home"
                    err_b_x = (x - home_x) * math.cos(yaw) + (y - home_y) * math.sin(yaw)
                    err_b_y = -(x - home_x) * math.sin(yaw) + (y - home_y) * math.cos(yaw)
                    if elapsed > HOVER_BEFORE_TRAJ and abs(z - TARGET_Z) < 0.35:
                        flight_state = "VLA_SEARCH"
                        print("\n🚀 爬升完成，切換至 VLA 尋找目標狀態...")

                elif flight_state == "VLA_SEARCH":
                    mode = "VLA_Search"
                    target_yaw += SEARCH_YAW_RATE * dt  # 原地旋轉尋找第一棵樹
                    err_b_x = err_b_y = 0.0             # 保持水平位置
                    if vla_state == "TRACKING":
                        flight_state = "VLA_TRACKING"
                        tree_count += 1
                        print(f"\n🎯 [VLA] 鎖定第 {tree_count} 棵樹！進入閉環動態追蹤。")

                elif flight_state == "VLA_TRACKING":
                    mode = "VLA_Track"
                    if vla_data:
                        # 2D 像素誤差映射至 3D 機身姿態誤差 (核心)
                        pixel_x = vla_data.get("pixel_err_x", 0.0)
                        pixel_y = vla_data.get("pixel_err_y", 0.0)
                        area_e = vla_data.get("area_err", 0.0)

                        # 1. Yaw：微調偏航角將目標置中
                        target_yaw = yaw - K_VLA_YAW * pixel_x
                        
                        # 2. X 軸 (Pitch)：依據面積誤差控制前後距離
                        err_b_x = K_VLA_FORWARD * area_e
                        
                        # 3. Y 軸 (Roll)：輔助側向對齊與抗風擾
                        err_b_y = K_VLA_LATERAL * pixel_x
                        
                        # 4. Z 軸：高度微調對準樹冠交界
                        target_z = max(0.5, min(TARGET_Z - K_VLA_ALTITUDE * pixel_y, 4.0))

                    if vla_state == "TRANSITION":
                        flight_state = "VLA_TRANSITION"
                        transition_start_time = curr_time
                        print(f"\n🔄 [VLA] 樹木巡檢完成，開始橫向平移尋找下一目標...")
                    
                    elif vla_state == "DONE":
                        flight_state = "RETURN_HOME"
                        transition_start_time = curr_time
                        print("\n🏁 [VLA] 收到 Done Token，全陣列樹木巡檢完成，啟動返航！")

                elif flight_state == "VLA_TRANSITION":
                    mode = "VLA_Trans"
                    # 在跨樹轉換時，不控制 Pitch (err_b_x = 0)，強制 Roll 輸出側向移動
                    err_b_x = 0.0 
                    err_b_y = -TRANSITION_SPEED * dt # 強制側風位移誤差，使控制器產生對應 Roll
                    
                    if vla_state == "TRACKING" and (curr_time - transition_start_time > 2.0):
                        flight_state = "VLA_TRACKING"
                        tree_count += 1
                        print(f"\n🎯 [VLA] 捕獲新特徵，鎖定第 {tree_count} 棵樹！")
                    elif vla_state == "DONE":
                        flight_state = "RETURN_HOME"
                        transition_start_time = curr_time

                elif flight_state == "RETURN_HOME":
                    mode = "Return_Home"
                    err_w_x, err_w_y = x - home_x, y - home_y
                    err_b_x = err_w_x * math.cos(yaw) + err_w_y * math.sin(yaw)
                    err_b_y = -err_w_x * math.sin(yaw) + err_w_y * math.cos(yaw)
                    dist_home = math.hypot(err_w_x, err_w_y)
                    if dist_home < 0.40:
                        flight_state = "LAND"
                        transition_start_time = curr_time
                        print("\n🛬 抵達 TurtleBot 4 (Home) 正上方，啟動降落...")

                elif flight_state == "LAND":
                    mode = "Land"
                    err_w_x, err_w_y = x - home_x, y - home_y
                    err_b_x = err_w_x * math.cos(yaw) + err_w_y * math.sin(yaw)
                    err_b_y = -err_w_x * math.sin(yaw) + err_w_y * math.cos(yaw)
                    t_land = curr_time - transition_start_time
                    target_z = max(0.0, TARGET_Z - LANDING_SPEED * t_land)

                # ==========================================
                # 控制量計算 (直接將機身誤差代入區間二型模糊歸屬函數)
                # ==========================================
                u_accel, trig_z = alt_ctrl.update(np.array([z, vz], dtype=float), target_z, dt)
                thrust_cmd = float(np.clip((HOVER_THRUST + u_accel * THRUST_SCALE) / max(math.cos(last_cmd_roll) * math.cos(last_cmd_pitch), 0.5), THR_MIN, THR_MAX))
                
                # 將映射好的機身體系誤差送入模糊網路控制器
                u_pitch, trig_x = etm_pos_x.update(err_b_x, err_b_vx, dt, curr_time)
                u_roll, trig_y = etm_pos_y.update(err_b_y, err_b_vy, dt, curr_time)

                target_roll = float(np.clip(-u_roll, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                target_pitch = float(np.clip(-u_pitch, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                last_cmd_roll, last_cmd_pitch = target_roll, target_pitch
                
                is_triggered = int(trig_x or trig_y or trig_z)
                if is_triggered or (curr_time - last_send_time > 0.4):
                    # 傳送 R, P, Yaw, Thrust
                    sock_send.sendto(struct.pack("<4fd", target_roll, target_pitch, target_yaw, thrust_cmd, curr_time), (PI_IP, UDP_SEND_PORT))
                    last_send_time = curr_time

                log_data.append([elapsed, x, y, z, err_b_x, err_b_y, target_z, is_triggered, flight_state])

                if curr_time - last_print_time >= 0.2:
                    last_print_time = curr_time
                    trig_str = "TX" if is_triggered else "--"
                    print(f"[{mode:12}] T:{elapsed:5.1f}s | XYZ({x:+4.1f},{y:+4.1f},{z:+4.1f}) | "
                          f"CMD R/P/Y({math.degrees(target_roll):+3.0f}°,{math.degrees(target_pitch):+3.0f}°,{math.degrees(target_yaw):+3.0f}°) | {trig_str}")

                if flight_state == "LAND" and target_z <= 0.0 and z < 0.25:
                    print("\n[Ground Controller] 已降落於移動載具，馬達全數鎖定！")
                    stop_msg = struct.pack("<4fd", 0.0, 0.0, 0.0, 0.0, time.time())
                    for _ in range(10):
                        sock_send.sendto(stop_msg, (PI_IP, UDP_SEND_PORT))
                        time.sleep(0.05)
                    break

    except KeyboardInterrupt:
        print("\n[Ground Controller] 使用者中斷任務")
    finally:
        sock_recv.close()
        sock_send.close()
        if log_data:
            fname = f"vla_generalization_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z', 'Err_B_X', 'Err_B_Y', 'Tgt_Z', 'Triggered', 'State'])
                writer.writerows(log_data)
            print(f"[Ground Controller] 紀錄已存至 {fname}")

if __name__ == "__main__":
    main()