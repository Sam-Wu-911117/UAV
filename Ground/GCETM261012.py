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
# 🎯 任務與路徑參數 (單排樹列單側外擺圓弧巡檢)
# ==========================================
TARGET_Z = 2.5                 # 基準巡檢高度 (公尺)
Z_VARIATION_AMP = 0.4          # 繞弧過程中的垂直高度微幅起伏 (公尺，獲取高低視角)
LANDING_SPEED = 0.5
HOVER_BEFORE_TRAJ = 3.0
TRANSITION_DURATION = 4.0      # 樹間直線過渡時間 (秒)

# 🌲 單排樹列定義 (樹列位於 Y=0 線上，無人機恆在 Y > 0 單側飛行)
# 格式: (樹木中心 X, 樹木中心 Y)
TREES = [
    (3.5, 0.0),   # 第 1 棵樹
    (7.5, 0.0),   # 第 2 棵樹
    (11.5, 0.0)   # 第 3 棵樹
]

# 弧線幾何參數
DEFAULT_ARC_RADIUS = 2.4       # 預設繞弧半徑 (公尺)
MIN_SAFE_Y = 1.2               # 單側防碰撞最小距離邊界 (公尺，保證永不穿越樹列)
ARC_DURATION = 6.0             # 繞單棵樹單側圓弧的巡航時間 (秒)
ARC_THETA_START = math.radians(35.0)   # 入弧角 (約 35 度，前斜視角)
ARC_THETA_END = math.radians(145.0)    # 出弧角 (約 145 度，後斜視角)

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
# 3. 軌跡幾何計算函式
# ==========================================
def min_jerk(t, T, p0, pf):
    t = max(0.0, min(t, T))
    tau = t / T
    tau3, tau4, tau5 = tau**3, tau**4, tau**5
    pos = p0 + (pf - p0) * (10*tau3 - 15*tau4 + 6*tau5)
    vel = (pf - p0) / T * (30*tau**2 - 60*tau3 + 30*tau4)
    acc = (pf - p0) / (T**2) * (60*tau - 180*tau**2 + 120*tau3)
    return pos, vel, acc

def calculate_tree_arc(tree_x, tree_y, radius, t_seg, T_total, z_base):
    """
    計算圍繞單棵樹單側的凸外擺圓弧 (X 方向單向朝前前進，Y 軸恆維持在單側正值)
    """
    t_clamped = max(0.0, min(t_seg, T_total))
    theta, dtheta, ddtheta = min_jerk(t_clamped, T_total, ARC_THETA_START, ARC_THETA_END)

    # 圓弧軌跡幾何 (半徑 R 沿著單側外擺)
    # x = tree_x - R * cos(theta), y = tree_y + R * sin(theta)
    x = tree_x - radius * math.cos(theta)
    y = max(tree_y + radius * math.sin(theta), MIN_SAFE_Y)

    # 速度前饋 (連鎖律)
    vx = radius * math.sin(theta) * dtheta
    vy = radius * math.cos(theta) * dtheta

    # 加速度前饋
    ax = radius * (math.cos(theta) * (dtheta**2) + math.sin(theta) * ddtheta)
    ay = radius * (-math.sin(theta) * (dtheta**2) + math.cos(theta) * ddtheta)

    # 高度微幅升降正弦波 (增加俯仰多角度取樣)
    tau = t_clamped / T_total
    z = z_base + Z_VARIATION_AMP * math.sin(2.0 * math.pi * tau)
    vz = Z_VARIATION_AMP * (2.0 * math.pi / T_total) * math.cos(2.0 * math.pi * tau)
    az = -Z_VARIATION_AMP * ((2.0 * math.pi / T_total)**2) * math.sin(2.0 * math.pi * tau)

    return x, y, z, vx, vy, vz, ax, ay, az

def get_arc_start_point(tree_x, tree_y, radius):
    """計算進入圓弧的起點座標 (前斜視角起點)"""
    x = tree_x - radius * math.cos(ARC_THETA_START)
    y = max(tree_y + radius * math.sin(ARC_THETA_START), MIN_SAFE_Y)
    return x, y

def request_vla_inference(req_id, tree_index):
    req_data = {
        "req_id": int(req_id),
        "tree_index": int(tree_index),
        "task": "ALIGN_TREE_SINGLE_SIDE",
        "timestamp": time.time()
    }
    with open(SHM_REQ_TEMP, 'w') as f:
        json.dump(req_data, f)
    os.replace(SHM_REQ_TEMP, SHM_REQ_PATH)

def read_vla_decision(expected_req_id):
    """讀取 VLA 決策：微調單側圓弧半徑與觀測高度"""
    if not os.path.exists(SHM_ACT_PATH):
        return None
    try:
        with open(SHM_ACT_PATH, 'r') as f:
            act = json.load(f)
        if act.get("event_id", -1) == expected_req_id and act.get("action") == "TREE_ARC_ALIGN":
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
    tree_idx = 0
    current_target_x = current_target_y = 0.0
    start_pos_x = start_pos_y = start_pos_z = 0.0
    transition_start_time = wait_vla_start_time = 0.0
    current_req_id = 0

    active_radius = DEFAULT_ARC_RADIUS
    active_tree_z = TARGET_Z

    print(f"\n[Ground Controller] 已啟用【單排樹列單側外擺圓弧巡檢】模式！共 {len(TREES)} 棵樹。")
    print(f"[Ground Controller] 固定相機配置：透過單向外擺圓弧獲得 前斜 ➔ 正側 ➔ 後斜 多角度觀測。")
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

                # --------------------------------------------------
                # 1. 爬升與初始懸停
                # --------------------------------------------------
                if flight_state == "HOVER_HOME":
                    mode = "Hover_Home"
                    target_x, target_y = home_x, home_y
                    if elapsed > HOVER_BEFORE_TRAJ and abs(z - TARGET_Z) < 0.35 and len(TREES) > 0:
                        flight_state = "APPROACH_ARC_ENTRY"
                        start_pos_x, start_pos_y = x, y
                        # 計算第 1 棵樹的入弧起點
                        tx, ty = TREES[tree_idx]
                        entry_x, entry_y = get_arc_start_point(home_x + tx, home_y + ty, DEFAULT_ARC_RADIUS)
                        current_target_x, current_target_y = entry_x, entry_y
                        transition_start_time = curr_time
                        print(f"\n🚀 啟程前往樹木 {tree_idx+1}/{len(TREES)} 入弧點: ({current_target_x:.2f}, {current_target_y:.2f})")

                # --------------------------------------------------
                # 2. 飛往外擺圓弧切入點 (單向平滑前進)
                # --------------------------------------------------
                elif flight_state == "APPROACH_ARC_ENTRY":
                    mode = f"Appr_T{tree_idx+1}"
                    t_seg = curr_time - transition_start_time
                    if t_seg < TRANSITION_DURATION:
                        target_x, tgt_vx, tgt_ax = min_jerk(t_seg, TRANSITION_DURATION, start_pos_x, current_target_x)
                        target_y, tgt_vy, tgt_ay = min_jerk(t_seg, TRANSITION_DURATION, start_pos_y, current_target_y)
                    else:
                        target_x, target_y = current_target_x, current_target_y
                        dist_err = math.hypot(x - current_target_x, y - current_target_y)
                        vel_xy = math.hypot(vx, vy)
                        
                        if (dist_err < 0.35 and vel_xy < 0.30) or (t_seg > TRANSITION_DURATION + 3.0):
                            flight_state = "WAIT_VLA_ARC"
                            wait_vla_start_time = curr_time
                            current_req_id += 1
                            request_vla_inference(current_req_id, tree_idx + 1)
                            print(f"\n📍 已抵達樹木 {tree_idx+1} 入弧前哨點！發送 VLA 幾何參數推論 (req_id={current_req_id})")

                # --------------------------------------------------
                # 3. 等待 VLA 微調圓弧半徑與相機對準參數
                # --------------------------------------------------
                elif flight_state == "WAIT_VLA_ARC":
                    mode = "Wait_VLA"
                    target_x, target_y = current_target_x, current_target_y
                    
                    vla_cmd = read_vla_decision(current_req_id)
                    vla_ready = False

                    if vla_cmd is not None:
                        # VLA 根據畫面微調圓弧半徑 (delta_radius) 與觀測高度 (delta_z)
                        delta_r = float(vla_cmd.get("delta_radius", 0.0))
                        delta_z = float(vla_cmd.get("delta_z", 0.0))
                        active_radius = float(np.clip(DEFAULT_ARC_RADIUS + delta_r, 1.8, 3.5))
                        active_tree_z = float(np.clip(TARGET_Z + delta_z, 1.5, 4.5))
                        vla_ready = True
                        print(f"🌲 [VLA 幾何確定] 半徑調整至 R={active_radius:.2f}m, 基準高度 Z={active_tree_z:.2f}m")

                    elif (curr_time - wait_vla_start_time) > 2.0:
                        # 逾時預設值
                        active_radius = DEFAULT_ARC_RADIUS
                        active_tree_z = TARGET_Z
                        vla_ready = True
                        print(f"⚠️ [VLA 回報逾時] 使用預設外擺半徑 R={active_radius:.2f}m, Z={active_tree_z:.2f}m")

                    if vla_ready:
                        flight_state = "EXECUTE_TREE_ARC"
                        transition_start_time = curr_time
                        print(f"🔄 開始執行單側外擺圓弧巡航 (單向朝 +X 前進，取得前斜/正側/後斜多視角)...")

                # --------------------------------------------------
                # 4. 執行單側凸外擺圓弧 (朝同方向移動 + 繞圓弧多角度取樣)
                # --------------------------------------------------
                elif flight_state == "EXECUTE_TREE_ARC":
                    t_seg = curr_time - transition_start_time
                    tx, ty = TREES[tree_idx]
                    center_x, center_y = home_x + tx, home_y + ty

                    target_x, target_y, tgt_z, tgt_vx, tgt_vy, tgt_z_vel, tgt_ax, tgt_ay, tgt_z_acc = calculate_tree_arc(
                        center_x, center_y, active_radius, t_seg, ARC_DURATION, active_tree_z
                    )

                    # 當前視角辨識 (前斜、正側、後斜)
                    tau = min(1.0, t_seg / ARC_DURATION)
                    if tau < 0.35:
                        mode = f"Arc_FrontOblique"
                    elif tau < 0.65:
                        mode = f"Arc_SideOrth"
                    else:
                        mode = f"Arc_RearOblique"

                    if t_seg >= ARC_DURATION:
                        tree_idx += 1
                        start_pos_x, start_pos_y, start_pos_z = x, y, z
                        transition_start_time = curr_time

                        if tree_idx < len(TREES):
                            # 前往下一個樹木的入弧起點 (載具持續朝 +X 前進，不回頭)
                            next_tx, next_ty = TREES[tree_idx]
                            current_target_x, current_target_y = get_arc_start_point(home_x + next_tx, home_y + next_ty, DEFAULT_ARC_RADIUS)
                            flight_state = "APPROACH_ARC_ENTRY"
                            print(f"\n✅ 樹木 {tree_idx} 單側多角度觀測完成！直飛下一棵樹入弧點: ({current_target_x:.2f}, {current_target_y:.2f})")
                        else:
                            flight_state = "RETURN_HOME"
                            print("\n🏁 整排樹列單側巡檢完成，啟動返航程序！")

                # --------------------------------------------------
                # 5. 返航程序
                # --------------------------------------------------
                elif flight_state == "RETURN_HOME":
                    mode = "Return_Home"
                    t_seg = curr_time - transition_start_time
                    if t_seg < TRANSITION_DURATION + 2.0:
                        target_x, tgt_vx, tgt_ax = min_jerk(t_seg, TRANSITION_DURATION + 2.0, start_pos_x, home_x)
                        target_y, tgt_vy, tgt_ay = min_jerk(t_seg, TRANSITION_DURATION + 2.0, start_pos_y, home_y)
                    else:
                        target_x, target_y = home_x, home_y
                        dist_home = math.hypot(x - home_x, y - home_y)
                        if dist_home < 0.40 or (t_seg > TRANSITION_DURATION + 5.0):
                            flight_state = "LAND"
                            transition_start_time = curr_time
                            print("\n🛬 已抵達起飛點正上方，開始垂直降落...")

                # --------------------------------------------------
                # 6. 垂直降落
                # --------------------------------------------------
                elif flight_state == "LAND":
                    mode = "Land"
                    target_x, target_y = home_x, home_y
                    t_land = curr_time - transition_start_time
                    tgt_z = max(0.0, TARGET_Z - LANDING_SPEED * t_land)

                # ==========================================
                # 控制量計算 (完全保留原有增益與歸屬函數架構)
                # ==========================================
                u_accel, trig_z = alt_ctrl.update(np.array([z, vz], dtype=float), tgt_z, dt)
                thrust_cmd = float(np.clip((HOVER_THRUST + u_accel * THRUST_SCALE) / max(math.cos(last_cmd_roll) * math.cos(last_cmd_pitch), 0.5), THR_MIN, THR_MAX))

                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                err_w_x, err_w_y = x - target_x, y - target_y
                err_w_vx, err_w_vy = vx - tgt_vx, vy - tgt_vy

                # 機體座標誤差轉換
                err_b_x = err_w_x * cos_y + err_w_y * sin_y
                err_b_y = -err_w_x * sin_y + err_w_y * cos_y
                err_b_vx = err_w_vx * cos_y + err_w_vy * sin_y
                err_b_vy = -err_w_vx * sin_y + err_w_vy * cos_y
                ff_b_ax = tgt_ax * cos_y + tgt_ay * sin_y
                ff_b_ay = -tgt_ax * sin_y + tgt_ay * cos_y
                
                # 區間二型模糊 ETM 核心計算
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
                    print(f"[{mode:17}] T:{elapsed:5.1f}s | XYZ({x:+4.1f},{y:+4.1f},{z:+4.1f}) | "
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
            fname = f"tree_arc_flight_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z', 'Tgt_X', 'Tgt_Y', 'Tgt_Z', 'Triggered', 'State'])
                writer.writerows(log_data)
            print(f"[Ground Controller] 紀錄已存至 {fname}")

if __name__ == "__main__":
    main()