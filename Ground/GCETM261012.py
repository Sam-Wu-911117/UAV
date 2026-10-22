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
# 1. 控制參數設定 (完全保留原有強健控制增益)
# ==========================================
UAV_MASS = 3.5
HOVER_FORCE = UAV_MASS * 9.81

OMEGA_ALT = np.array([[11.2683, 16.5707], [16.5707, 25.0074]], dtype=float)
F1_ALT = np.array([-8.1979, -12.7244], dtype=float)
SIGMA_ALT = 0.00
AR_ALT = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)
HOVER_THRUST = 0.65
THRUST_SCALE = 0.045
THR_MIN, THR_MAX = 0.10, 0.90

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
# 🎯 任務與路徑參數 (太陽能板垂直對準巡檢)
# ==========================================
TARGET_Z = 6.5                 # 基準巡航高度 (可依太陽能板場域修改)
LANDING_SPEED = 0.5
HOVER_BEFORE_TRAJ = 3.0
TRANSITION_DURATION = 5.0      # 航點間平滑飛行時間 (5 秒)

# ☀️ 太陽能板相機垂直對準專屬參數
SOLAR_ALIGN_DURATION = 2.5     # 根據 VLA 調整位置對準面板的平滑移動時間 (2.5 秒)
SOLAR_MAX_OFFSET_XY = 1.5      # 單次 XY 軸最大位置修正限制 (公尺，防止異常數值爆衝)
SOLAR_MIN_Z, SOLAR_MAX_Z = 2.0, 8.0 # 高度調整安全邊界 (公尺)
SOLAR_HOVER_TIME = 3.0         # 對準後固定懸停拍攝/檢視時間 (3.0 秒)

TARGET_POINTS = [
    (3.0, 1.10),  # 太陽能板觀測點 1 (粗略座標)
    (5.5, 2.10)   # 太陽能板觀測點 2 (粗略座標)
]

SHM_REQ_TEMP = '/dev/shm/uav_req_tmp.json'
SHM_REQ_PATH = '/dev/shm/uav_req.json'
SHM_ACT_PATH = '/dev/shm/uav_action.json'

# ==========================================
# 2. ETM 控制器 (完全維持原有增益與歸屬函數架構)
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

    def update(self, current_state, target_val, dt, current_time):
        r_input = np.array([0.0, 4.0 * target_val], dtype=float)
        self.ref_state += (self.Ar @ self.ref_state + r_input) * dt
        e_trk = current_state - self.ref_state
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
        # 依據追蹤誤差計算模糊歸屬函數權重 w2
        w2 = float(np.clip(abs(float(e_trk[0])) / POS_ERR_MAX, 0.0, 1.0))
        F_fuzzy = (1.0 - w2) * self.F1 + w2 * self.F2
        alpha = max(min(1.0, float(np.linalg.norm(e_ctrl)) / SOFT_ERR_XY), ALPHA_MIN)
        u_raw = float(alpha * (F_fuzzy @ e_ctrl) * GAIN_SCALE_XY)
        du = (u_raw - self.prev_final_u) / dt
        u_final = self.prev_final_u + np.sign(du) * RATE_LIMIT_XY * dt if abs(du) > RATE_LIMIT_XY else u_raw
        self.last_control_u = self.prev_final_u = float(u_final)
        return self.last_control_u, triggered

# ==========================================
# 3. 輔助函式
# ==========================================
def min_jerk(t, T, p0, pf):
    t = max(0.0, min(t, T))
    tau = t / T
    tau3, tau4, tau5 = tau**3, tau**4, tau**5
    pos = p0 + (pf - p0) * (10*tau3 - 15*tau4 + 6*tau5)
    vel = (pf - p0) / T * (30*tau**2 - 60*tau3 + 30*tau4)
    acc = (pf - p0) / (T**2) * (60*tau - 180*tau**2 + 30*4*tau3)
    return pos, vel, acc

def request_vla_inference(req_id, wp_index):
    req_data = {
        "req_id": int(req_id),
        "wp_index": int(wp_index),
        "timestamp": time.time()
    }
    with open(SHM_REQ_TEMP, 'w') as f:
        json.dump(req_data, f)
    os.replace(SHM_REQ_TEMP, SHM_REQ_PATH)

def read_vla_decision(expected_req_id):
    """讀取 VLA 決策：支援 SOLAR_ALIGN (太陽能板相機垂直對準)"""
    if not os.path.exists(SHM_ACT_PATH):
        return None
    try:
        with open(SHM_ACT_PATH, 'r') as f:
            act = json.load(f)
        if act.get("event_id", -1) == expected_req_id and act.get("action") == "SOLAR_ALIGN":
            return act
    except Exception:
        pass
    return None

# ==========================================
# 4. 主程式
# ==========================================
def main():
    for p in [SHM_REQ_PATH, SHM_ACT_PATH]:
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    PI_IP = input("請輸入樹莓派 IP: ").strip()
    UDP_LISTEN_PORT, UDP_SEND_PORT = 5006, 5005

    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", UDP_LISTEN_PORT))
    sock_recv.settimeout(0.5)
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    alt_ctrl = Altitude_ETM_Controller()
    etm_pos_x = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS)
    etm_pos_y = Fuzzy_ETM_Core(OMEGA_POS, F1_POS, F2_POS, AR_POS)
    
    log_data = []
    home_initialized = False
    home_x = home_y = 0.0
    last_cmd_roll, last_cmd_pitch = 0.0, 0.0

    flight_state = "HOVER_HOME"
    wp_index = 0
    current_target_x = current_target_y = 0.0
    start_pos_x = start_pos_y = start_pos_z = 0.0
    transition_start_time = wait_vla_start_time = 0.0
    hover_remaining = 0.0
    current_req_id = 0

    align_target_x = align_target_y = 0.0
    align_target_z = TARGET_Z

    print(f"\n[Ground Controller] 已啟用【VLA 太陽能板垂直對準巡檢】模式！共有 {len(TARGET_POINTS)} 個觀測點。")
    print(f"[Ground Controller] 基準巡航高度鎖定: Z = {TARGET_Z} 公尺")
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
                    start_time = prev_time = curr_time
                    home_initialized = True
                    print(f"✅ Home 鎖定: X={home_x:.2f}, Y={home_y:.2f}。準備爬升至 {TARGET_Z}m...")

                dt = min(max(curr_time - prev_time, DT_MIN), DT_MAX)
                prev_time = curr_time
                elapsed = curr_time - start_time

                tgt_vx = tgt_vy = tgt_ax = tgt_ay = 0.0
                tgt_z = TARGET_Z

                # 1. 爬升與初始懸停
                if flight_state == "HOVER_HOME":
                    mode = "Hover_Home"
                    target_x, target_y = home_x, home_y
                    if elapsed > HOVER_BEFORE_TRAJ and abs(z - TARGET_Z) < 0.35 and len(TARGET_POINTS) > 0:
                        flight_state = "GO_WAYPOINT"
                        start_pos_x, start_pos_y = x, y
                        current_target_x = home_x + TARGET_POINTS[wp_index][0]
                        current_target_y = home_y + TARGET_POINTS[wp_index][1]
                        transition_start_time = curr_time
                        print(f"\n🚀 啟程前往太陽能板觀測點 {wp_index+1}/{len(TARGET_POINTS)}: ({current_target_x:.2f}, {current_target_y:.2f})")

                # 2. 飛往任務航點
                elif flight_state == "GO_WAYPOINT":
                    mode = f"Go_WP_{wp_index+1}"
                    t_seg = curr_time - transition_start_time
                    if t_seg < TRANSITION_DURATION:
                        target_x, tgt_vx, tgt_ax = min_jerk(t_seg, TRANSITION_DURATION, start_pos_x, current_target_x)
                        target_y, tgt_vy, tgt_ay = min_jerk(t_seg, TRANSITION_DURATION, start_pos_y, current_target_y)
                    else:
                        target_x, target_y = current_target_x, current_target_y
                        dist_err = math.hypot(x - current_target_x, y - current_target_y)
                        vel_xy = math.hypot(vx, vy)
                        
                        if (dist_err < 0.40 and vel_xy < 0.35) or (t_seg > TRANSITION_DURATION + 4.0):
                            flight_state = "WAIT_VLA"
                            wait_vla_start_time = curr_time
                            current_req_id += 1
                            request_vla_inference(current_req_id, wp_index + 1)
                            print(f"\n📍 已抵達觀測點上方 {wp_index+1} (誤差:{dist_err:.2f}m)！發送 VLA 視覺對準請求 (req_id={current_req_id})")

                # 3. 等待 VLA 太陽能板對準決策
                elif flight_state == "WAIT_VLA":
                    mode = "Wait_VLA"
                    target_x, target_y = current_target_x, current_target_y
                    
                    vla_cmd = read_vla_decision(current_req_id)
                    if vla_cmd is not None:
                        # 接收 VLA 計算之機體座標系修正量 (單位: 公尺)
                        # offset_x: 機頭前後方向修正 (+為往前飛, -為往後退)
                        # offset_y: 機身左右方向修正 (+為往左移, -為往右移)
                        # offset_z: 垂直高度修正 (+為往上爬升, -為往下降低)
                        dx_body = float(np.clip(float(vla_cmd.get("offset_x", 0.0)), -SOLAR_MAX_OFFSET_XY, SOLAR_MAX_OFFSET_XY))
                        dy_body = float(np.clip(float(vla_cmd.get("offset_y", 0.0)), -SOLAR_MAX_OFFSET_XY, SOLAR_MAX_OFFSET_XY))
                        dz_body = float(vla_cmd.get("offset_z", 0.0))
                        
                        # 將機體座標修正量透過當前偏航角 (yaw) 轉換至世界座標系
                        align_target_x = x + (dx_body * math.cos(yaw) - dy_body * math.sin(yaw))
                        align_target_y = y + (dx_body * math.sin(yaw) + dy_body * math.cos(yaw))
                        align_target_z = float(np.clip(TARGET_Z + dz_body, SOLAR_MIN_Z, SOLAR_MAX_Z))
                        hover_remaining = float(vla_cmd.get("hover_time", SOLAR_HOVER_TIME))
                        
                        start_pos_x, start_pos_y, start_pos_z = x, y, z
                        transition_start_time = curr_time
                        flight_state = "SOLAR_ALIGN_MOVE"
                        print(f"☀️ [VLA 對準觸發] 機體修正 (dx={dx_body:+.2f}m, dy={dy_body:+.2f}m, dz={dz_body:+.2f}m) "
                              f"➔ 移至正上方 ({align_target_x:.2f}, {align_target_y:.2f}, Z={align_target_z:.2f})")

                    elif (curr_time - wait_vla_start_time) > 4.0:
                        # 若 VLA 逾時未回傳，維持當前位置與基準高度直接進行拍攝巡檢
                        align_target_x, align_target_y = current_target_x, current_target_y
                        align_target_z = TARGET_Z
                        hover_remaining = SOLAR_HOVER_TIME
                        start_pos_x, start_pos_y, start_pos_z = x, y, z
                        transition_start_time = curr_time
                        flight_state = "SOLAR_ALIGN_MOVE"
                        print(f"⚠️ [VLA 回報逾時] 維持當前座標 ({align_target_x:.2f}, {align_target_y:.2f}, Z={align_target_z:.2f}) 懸停 {hover_remaining:.1f} 秒！")

                # ==========================================
                # ☀️ 太陽能板垂直對準三階段子狀態機
                # ==========================================
                # 階段 1：三軸同步平滑移動，讓相機正對太陽能板中心
                elif flight_state == "SOLAR_ALIGN_MOVE":
                    mode = "Solar_Align"
                    t_seg = curr_time - transition_start_time
                    if t_seg < SOLAR_ALIGN_DURATION:
                        target_x, tgt_vx, tgt_ax = min_jerk(t_seg, SOLAR_ALIGN_DURATION, start_pos_x, align_target_x)
                        target_y, tgt_vy, tgt_ay = min_jerk(t_seg, SOLAR_ALIGN_DURATION, start_pos_y, align_target_y)
                        tgt_z, _, _ = min_jerk(t_seg, SOLAR_ALIGN_DURATION, start_pos_z, align_target_z)
                    else:
                        target_x, target_y, tgt_z = align_target_x, align_target_y, align_target_z
                        dist_err = math.hypot(x - align_target_x, y - align_target_y)
                        z_err = abs(z - align_target_z)
                        if (dist_err < 0.25 and z_err < 0.25) or (t_seg > SOLAR_ALIGN_DURATION + 2.0):
                            flight_state = "SOLAR_HOVER_INSPECT"
                            print(f"📸 相機已垂直對準太陽能板 (XY誤差:{dist_err:.2f}m, Z={z:.2f}m)，穩定懸停 {hover_remaining:.1f} 秒進行巡檢！")

                # 階段 2：定點穩定懸停巡檢 (此時機身 Roll/Pitch 趨近 0 度，確保相機畫面垂直朝下)
                elif flight_state == "SOLAR_HOVER_INSPECT":
                    mode = f"Solar({hover_remaining:.1f}s)"
                    target_x, target_y = align_target_x, align_target_y
                    tgt_z = align_target_z
                    hover_remaining -= dt
                    if hover_remaining <= 0.0:
                        flight_state = "SOLAR_ASCEND_Z"
                        start_pos_z = z
                        transition_start_time = curr_time
                        print("✅ 太陽能板巡檢完成！準備恢復基準巡航高度...")

                # 階段 3：恢復基準高度並前往下一處太陽能板
                elif flight_state == "SOLAR_ASCEND_Z":
                    mode = "Solar_Asc"
                    target_x, target_y = align_target_x, align_target_y
                    t_seg = curr_time - transition_start_time
                    if t_seg < SOLAR_ALIGN_DURATION:
                        tgt_z, _, _ = min_jerk(t_seg, SOLAR_ALIGN_DURATION, start_pos_z, TARGET_Z)
                    else:
                        tgt_z = TARGET_Z
                        if abs(z - TARGET_Z) < 0.35 or (t_seg > SOLAR_ALIGN_DURATION + 2.0):
                            wp_index += 1
                            start_pos_x, start_pos_y = x, y
                            transition_start_time = curr_time
                            if wp_index < len(TARGET_POINTS):
                                current_target_x = home_x + TARGET_POINTS[wp_index][0]
                                current_target_y = home_y + TARGET_POINTS[wp_index][1]
                                flight_state = "GO_WAYPOINT"
                                print(f"\n🚀 已恢復基準高度，前往下一太陽能板觀測點 {wp_index+1}/{len(TARGET_POINTS)}: ({current_target_x:.2f}, {current_target_y:.2f})")
                            else:
                                flight_state = "RETURN_HOME"
                                print("\n🏁 所有太陽能板均已巡檢完畢，啟動返航程序！")

                # 4. 返航回 Home 點
                elif flight_state == "RETURN_HOME":
                    mode = "Return_Home"
                    t_seg = curr_time - transition_start_time
                    if t_seg < TRANSITION_DURATION:
                        target_x, tgt_vx, tgt_ax = min_jerk(t_seg, TRANSITION_DURATION, start_pos_x, home_x)
                        target_y, tgt_vy, tgt_ay = min_jerk(t_seg, TRANSITION_DURATION, start_pos_y, home_y)
                    else:
                        target_x, target_y = home_x, home_y
                        dist_home = math.hypot(x - home_x, y - home_y)
                        if dist_home < 0.40 or (t_seg > TRANSITION_DURATION + 3.0):
                            flight_state = "LAND"
                            transition_start_time = curr_time
                            print("\n🛬 已安全返航至起飛點正上方，開始垂直平穩降落...")

                # 5. 降落
                elif flight_state == "LAND":
                    mode = "Land"
                    target_x, target_y = home_x, home_y
                    t_land = curr_time - transition_start_time
                    tgt_z = max(0.0, TARGET_Z - LANDING_SPEED * t_land)

                # 控制量計算 (完全保留原有增益與歸屬函數架構)
                u_accel, trig_z = alt_ctrl.update(np.array([z, vz], dtype=float), tgt_z, dt)
                thrust_cmd = float(np.clip((HOVER_THRUST + u_accel * THRUST_SCALE) / max(math.cos(last_cmd_roll) * math.cos(last_cmd_pitch), 0.5), THR_MIN, THR_MAX))

                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                err_w_x, err_w_y = x - target_x, y - target_y
                err_w_vx, err_w_vy = vx - tgt_vx, vy - tgt_vy

                err_b_x = err_w_x * cos_y + err_w_y * sin_y
                err_b_y = -err_w_x * sin_y + err_w_y * cos_y
                err_b_vx = err_w_vx * cos_y + err_w_vy * sin_y
                err_b_vy = -err_w_vx * sin_y + err_w_vy * cos_y
                ff_b_ax = tgt_ax * cos_y + tgt_ay * sin_y
                ff_b_ay = -tgt_ax * sin_y + tgt_ay * cos_y
                
                u_pitch, trig_x = etm_pos_x.update(np.array([err_b_x, err_b_vx], dtype=float), 0.0, dt, curr_time)
                u_roll, trig_y = etm_pos_y.update(np.array([err_b_y, err_b_vy], dtype=float), 0.0, dt, curr_time)

                target_roll = float(np.clip(-u_roll + ff_b_ay * FF_ACCEL_GAIN, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                target_pitch = float(np.clip(-u_pitch - ff_b_ax * FF_ACCEL_GAIN, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                last_cmd_roll, last_cmd_pitch = target_roll, target_pitch
                
                is_triggered = int(trig_x or trig_y or trig_z)
                if is_triggered or (curr_time - last_send_time > 0.4):
                    sock_send.sendto(struct.pack("<4fd", target_roll, target_pitch, 0.0, thrust_cmd, curr_time), (PI_IP, UDP_SEND_PORT))
                    last_send_time = curr_time

                log_data.append([elapsed, x, y, z, target_x, target_y, tgt_z, is_triggered, flight_state])

                if curr_time - last_print_time >= 0.2:
                    last_print_time = curr_time
                    trig_str = "TX" if is_triggered else "--"
                    print(f"[{mode:12}] T:{elapsed:5.1f}s | XYZ({x:+4.1f},{y:+4.1f},{z:+4.1f}) | "
                          f"CMD R/P({math.degrees(target_roll):+4.0f}°,{math.degrees(target_pitch):+4.0f}°) | {trig_str}")

                if flight_state == "LAND" and tgt_z <= 0.0 and z < 0.25:
                    print("\n[Ground Controller] 已落地，馬達全數鎖定！")
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
            fname = f"solar_flight_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z', 'Tgt_X', 'Tgt_Y', 'Tgt_Z', 'Triggered', 'State'])
                writer.writerows(log_data)
            print(f"[Ground Controller] 紀錄已存至 {fname}")

if __name__ == "__main__":
    main()