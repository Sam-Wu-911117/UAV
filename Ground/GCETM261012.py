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
# 1. 控制參數設定
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
# 🎯 任務與路徑參數 (太陽能板 5.0m + 上方 1.5m = 6.5m)
# ==========================================
TARGET_Z = 6.5
LANDING_SPEED = 0.5
HOVER_BEFORE_TRAJ = 3.0
TRANSITION_DURATION = 4.0

KNOWN_DEFECTS = [
    (3.0, 1.10),  # 髒污區 1：第一排面板 (落葉群)
    (5.5, 2.10)   # 髒污區 2：第二排面板 (大面積泥沙)
]

SHM_REQ_TEMP = '/dev/shm/uav_req_tmp.json'
SHM_REQ_PATH = '/dev/shm/uav_req.json'
SHM_ACT_PATH = '/dev/shm/uav_action.json'

# ==========================================
# 2. ETM 控制器
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
        w2 = float(np.clip(abs(float(e_trk[0])) / POS_ERR_MAX, 0.0, 1.0))
        F_fuzzy = (1.0 - w2) * self.F1 + w2 * self.F2
        alpha = max(min(1.0, float(np.linalg.norm(e_ctrl)) / SOFT_ERR_XY), ALPHA_MIN)
        u_raw = float(alpha * (F_fuzzy @ e_ctrl) * GAIN_SCALE_XY)
        du = (u_raw - self.prev_final_u) / dt
        u_final = self.prev_final_u + np.sign(du) * RATE_LIMIT_XY * dt if abs(du) > RATE_LIMIT_XY else u_raw
        self.last_control_u = self.prev_final_u = float(u_final)
        return self.last_control_u, triggered

# ==========================================
# 3. 輔助函式 (按需觸發拍照與讀取結果)
# ==========================================
def min_jerk(t, T, p0, pf):
    t = max(0.0, min(t, T))
    tau = t / T
    tau3, tau4, tau5 = tau**3, tau**4, tau**5
    pos = p0 + (pf - p0) * (10*tau3 - 15*tau4 + 6*tau5)
    vel = (pf - p0) / T * (30*tau**2 - 60*tau3 + 30*tau4)
    acc = (pf - p0) / (T**2) * (60*tau - 180*tau**2 + 120*tau3)
    return pos, vel, acc

def request_vla_inference(req_id, wp_index):
    """當抵達髒污點時，向終端機 2 發出單次拍照與 VLA 推論請求"""
    req_data = {
        "req_id": int(req_id),
        "wp_index": int(wp_index),
        "timestamp": time.time()
    }
    with open(SHM_REQ_TEMP, 'w') as f:
        json.dump(req_data, f)
    os.replace(SHM_REQ_TEMP, SHM_REQ_PATH)

def read_vla_cleaning_duration(expected_req_id):
    """只讀取與當前 req_id 完全匹配的 VLA 推論結果"""
    if not os.path.exists(SHM_ACT_PATH):
        return None
    try:
        with open(SHM_ACT_PATH, 'r') as f:
            act = json.load(f)
        if act.get("action") == "CLEAN_DURATION" and act.get("event_id", -1) == expected_req_id:
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
    start_pos_x = start_pos_y = 0.0
    transition_start_time = wait_vla_start_time = 0.0
    clean_hover_remaining = 0.0
    current_req_id = 0

    print(f"\n[Ground Controller] 已啟用【按需觸發 VLA 推論】模式！共有 {len(KNOWN_DEFECTS)} 個髒污點待清。")
    print(f"[Ground Controller] 高度已設定為: Z = {TARGET_Z} 公尺 (面板 5.0m + 上方 1.5m)")
    print(f"[Ground Controller] 等待無人機心跳包...")

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

                if flight_state == "HOVER_HOME":
                    mode = "Hover_Home"
                    target_x, target_y = home_x, home_y
                    if elapsed > HOVER_BEFORE_TRAJ and len(KNOWN_DEFECTS) > 0:
                        flight_state = "GO_WAYPOINT"
                        start_pos_x, start_pos_y = x, y
                        current_target_x = home_x + KNOWN_DEFECTS[wp_index][0]
                        current_target_y = home_y + KNOWN_DEFECTS[wp_index][1]
                        transition_start_time = curr_time
                        print(f"\n🚀 啟程前往髒污區 {wp_index+1}/{len(KNOWN_DEFECTS)}: ({current_target_x:.2f}, {current_target_y:.2f}) [飛行途中不發送影像]")

                elif flight_state == "GO_WAYPOINT":
                    mode = f"Go_WP_{wp_index+1}"
                    t_seg = curr_time - transition_start_time
                    if t_seg < TRANSITION_DURATION:
                        target_x, tgt_vx, tgt_ax = min_jerk(t_seg, TRANSITION_DURATION, start_pos_x, current_target_x)
                        target_y, tgt_vy, tgt_ay = min_jerk(t_seg, TRANSITION_DURATION, start_pos_y, current_target_y)
                    else:
                        target_x, target_y = current_target_x, current_target_y
                        flight_state = "WAIT_VLA"
                        wait_vla_start_time = curr_time
                        
                        # 抵達髒污點！發出單次拍照與 VLA 推論請求
                        current_req_id += 1
                        request_vla_inference(current_req_id, wp_index + 1)
                        print(f"\n📍 已抵達髒污區 {wp_index+1}！已發送單次拍照請求 (req_id={current_req_id})，等待 SmolVLA 推論...")

                elif flight_state == "WAIT_VLA":
                    mode = "Wait_VLA"
                    target_x, target_y = current_target_x, current_target_y
                    
                    vla_cmd = read_vla_cleaning_duration(current_req_id)
                    if vla_cmd is not None:
                        clean_hover_remaining = float(vla_cmd["hover_duration"])
                        flight_state = "CLEAN_HOVER"
                        print(f"🧽 [收到 SmolVLA 單次推論回報] 髒污指標 {vla_cmd.get('dirt_score', 0.0)*100:.1f}% ➔ 開始定點清洗 {clean_hover_remaining:.1f} 秒！")
                    elif (curr_time - wait_vla_start_time) > 5.0:
                        clean_hover_remaining = 6.0
                        flight_state = "CLEAN_HOVER"
                        print("⚠️ [SmolVLA 等待超時] 啟動預設保底清洗 6.0 秒！")

                elif flight_state == "CLEAN_HOVER":
                    mode = f"Clean({clean_hover_remaining:.1f}s)"
                    target_x, target_y = current_target_x, current_target_y
                    clean_hover_remaining -= dt
                    
                    if clean_hover_remaining <= 0.0:
                        print(f"✅ 髒污區 {wp_index+1} 清洗完成！")
                        wp_index += 1
                        start_pos_x, start_pos_y = x, y
                        transition_start_time = curr_time
                        
                        if wp_index < len(KNOWN_DEFECTS):
                            current_target_x = home_x + KNOWN_DEFECTS[wp_index][0]
                            current_target_y = home_y + KNOWN_DEFECTS[wp_index][1]
                            flight_state = "GO_WAYPOINT"
                            print(f"\n🚀 繼續前往髒污區 {wp_index+1}/{len(KNOWN_DEFECTS)}: ({current_target_x:.2f}, {current_target_y:.2f}) [飛行途中不發送影像]")
                        else:
                            flight_state = "RETURN_HOME"
                            print("\n🏁 所有髒污點均已清洗完畢，準備返航...")

                elif flight_state == "RETURN_HOME":
                    mode = "Return_Home"
                    t_seg = curr_time - transition_start_time
                    if t_seg < TRANSITION_DURATION:
                        target_x, tgt_vx, tgt_ax = min_jerk(t_seg, TRANSITION_DURATION, start_pos_x, home_x)
                        target_y, tgt_vy, tgt_ay = min_jerk(t_seg, TRANSITION_DURATION, start_pos_y, home_y)
                    else:
                        target_x, target_y = home_x, home_y
                        flight_state = "LAND"
                        transition_start_time = curr_time
                        print("\n🛬 抵達起飛點，啟動平滑降落程序...")

                elif flight_state == "LAND":
                    mode = "Land"
                    target_x, target_y = home_x, home_y
                    t_land = curr_time - transition_start_time
                    tgt_z = max(0.0, TARGET_Z - LANDING_SPEED * t_land)

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

                if flight_state == "LAND" and tgt_z <= 0.0 and z < 0.2:
                    print("\n[Ground Controller] 降落完成，馬達鎖定！")
                    stop_msg = struct.pack("<4fd", 0.0, 0.0, 0.0, 0.0, time.time())
                    for _ in range(10):
                        sock_send.sendto(stop_msg, (PI_IP, UDP_SEND_PORT))
                        time.sleep(0.05)
                    break

    except KeyboardInterrupt:
        print("\n[Ground Controller] 使用者中斷")
    finally:
        sock_recv.close()
        sock_send.close()
        if log_data:
            fname = f"ondemand_vla_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z', 'Tgt_X', 'Tgt_Y', 'Tgt_Z', 'Triggered', 'State'])
                writer.writerows(log_data)
            print(f"[Ground Controller] 飛行紀錄已儲存至 {fname}")

if __name__ == "__main__":
    main()