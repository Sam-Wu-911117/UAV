#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import socket
import struct
import time
from pathlib import Path

import numpy as np

# =========================================================
# 網路與通訊參數
# =========================================================
PI_IP = input("請輸入樹莓派 IP (例如 192.168.1.101): ").strip()
UDP_SEND_PORT = 5005   # Ground -> Pi
UDP_RECV_PORT = 5006   # Pi -> Ground

# =========================================================
# 模式設定
# =========================================================
ENABLE_FIGURE8 = True

# 測試模式：
FIXED_CMD_TEST = False
Z_ONLY = False
XY_ONLY = False

# 固定命令測試值（無槳專用）
FIXED_TEST_ROLL = 0.1
FIXED_TEST_PITCH = 0.0
FIXED_TEST_YAW_RATE = 0.0
FIXED_TEST_THRUST = 0.4

# =========================================================
# 安全與實驗參數 (已對齊 MATLAB init_params.m)
# =========================================================
TARGET_Z = 5.0

SPOOL_TIME = 3.0
TAKEOFF_TIME = 3.0

HOVER_THRUST = 0.52
THRUST_SCALE = 0.045
ROLL_PITCH_LIMIT = 0.6  # 對齊 Ctrl.ROLL_PITCH_LIMIT
MAX_SEND_INTERVAL = 0.1

STATE_TIMEOUT = 1.0
DT_MIN = 0.01
DT_MAX = 0.03

# 高度控制參數
OMEGA_ALT = np.array([[11.2683, 16.5707], [16.5707, 25.0074]], dtype=float)
F1_ALT = np.array([-8.1979, -12.7244], dtype=float)
SIGMA_ALT = 0.01  # 對齊 Ctrl.SIGMA_ALT
AR_ALT = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)

THR_MIN = 0.10
THR_MAX = 0.90
ALT_SOFT_CEIL = 6.0
ALT_HARD_CEIL = 8.0

# XY 控制參數
OMEGA_POS = np.array([[282.0784, 501.9677], [501.9677, 893.5577]], dtype=float)
F1_POS = np.array([-2.4604, -4.4041], dtype=float)
F2_POS = np.array([-2.9273, -5.2102], dtype=float)
AR_POS = np.array([[0.0, 1.0], [-4.0, -4.0]], dtype=float)
SIGMA_POS = 0.05  # 對齊 Ctrl.SIGMA_POS

POS_ERR_MAX = 1.0
RATE_LIMIT_XY = 3.0
GAIN_SCALE_XY = 0.03
MIN_TRIGGER_INTERVAL = 0.05
TAU_FILTER_XY = 0.15
POS_DEADZONE_XY = 0.001  # 對齊 Ctrl.POS_DEADZONE_XY
VEL_DEADZONE_XY = 0.01   # 對齊 Ctrl.VEL_DEADZONE_XY
SOFT_ERR_XY = 0.01       # 對齊 Ctrl.SOFT_ERR_XY
ALPHA_MIN = 0.15
FF_ACCEL_GAIN = 0.10     # 對齊 Ctrl.FF_ACCEL_GAIN

XY_ENABLE_ALTITUDE = 0.30
LOW_ALT_XY_GAIN = 0.30
LOW_ALT_ROLL_PITCH_LIMIT = 0.03

# 軌跡參數 (對齊 MATLAB Traj 結構)
HOVER_BEFORE_TRAJ = 3.0
FIG8_A = 10.0
FIG8_B = 10.0
FIG8_OMEGA = 0.052
FIG8_PERIOD = 2.0 * math.pi / FIG8_OMEGA
FIG8_LOOPS = 1
TOTAL_FIG8_TIME = FIG8_PERIOD * FIG8_LOOPS
LANDING_SPEED = 0.5

# =========================================================
# Logging
# =========================================================
ENABLE_LOG = True
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"etm_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# =========================================================
# 控制器類別
# =========================================================
class Altitude_ETM_Controller:
    def __init__(self):
        self.last_sent_state = np.zeros(2, dtype=float)
        self.last_control_u = 0.0
        self.ref_state = np.zeros(2, dtype=float)
        self.prev_time = time.time()
        self.first_run = True

    def compute_control(self, current_state, target_height, dt):
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
            u = float(np.clip(u, -8.0, 8.0)) # 對齊 MATLAB 內部 Clip
            self.last_control_u = u
        else:
            u = self.last_control_u

        return u, triggered, e_trk, term_net, term_trk


class Fuzzy_ETM_Core:
    def __init__(self, omega, f1, f2, ar, name="Sys",
                 rate_limit=1.5, gain_scale=1.0,
                 pos_deadzone=0.03, vel_deadzone=0.05, soft_err=0.3):
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
        return w1 * self.F1 + w2 * self.F2

    def apply_deadzone(self, e):
        e_out = e.copy()
        if abs(e_out[0]) < self.pos_deadzone:
            e_out[0] = 0.0
        if abs(e_out[1]) < self.vel_deadzone:
            e_out[1] = 0.0
        return e_out

    def update(self, current_state, target_val, dt, current_time):
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
            ((current_time - self.last_trigger_time) >= MIN_TRIGGER_INTERVAL)
        ):
            triggered = True
            self.first_run = False
            self.last_trigger_time = current_time
            self.last_sent_error = e_trk.copy()

        self.filtered_error += (dt / TAU_FILTER_XY) * (self.last_sent_error - self.filtered_error)
        e_ctrl = self.apply_deadzone(self.filtered_error)

        F_fuzzy = self.get_fuzzy_gain(e_trk)
        
        e_norm = float(np.linalg.norm(e_ctrl))
        alpha = min(1.0, e_norm / self.soft_err) if self.soft_err > 1e-9 else 1.0
        alpha = max(alpha, ALPHA_MIN)

        u_raw = float(alpha * (F_fuzzy @ e_ctrl) * self.gain_scale)
        du = (u_raw - self.prev_final_u) / dt

        if abs(du) > self.rate_limit:
            u_final = self.prev_final_u + np.sign(du) * self.rate_limit * dt
        else:
            u_final = u_raw

        self.last_control_u = float(u_final)
        self.prev_final_u = self.last_control_u

        return self.last_control_u, triggered, e_trk, e_ctrl, term_net, term_trk


# =========================================================
# 軌跡產生 (對齊 MATLAB generate_trajectory)
# =========================================================
def generate_trajectory(home_x, home_y, flight_time):
    if not ENABLE_FIGURE8:
        return home_x, home_y, TARGET_Z, 0.0, 0.0, 0.0, 0.0

    t_hover_end = HOVER_BEFORE_TRAJ
    t_fig8_end = t_hover_end + TOTAL_FIG8_TIME

    if flight_time < t_hover_end:
        return home_x, home_y, TARGET_Z, 0.0, 0.0, 0.0, 0.0
        
    elif flight_time < t_fig8_end:
        t_f = flight_time - t_hover_end
        A = FIG8_A
        B = FIG8_B
        w = FIG8_OMEGA
        target_x = home_x + A * math.sin(w * t_f)
        target_y = home_y + B * math.sin(w * t_f) * math.cos(w * t_f)
        target_vx = A * w * math.cos(w * t_f)
        target_vy = B * w * math.cos(2.0 * w * t_f)
        target_ax = -A * (w**2) * math.sin(w * t_f)
        target_ay = -2.0 * B * (w**2) * math.sin(2.0 * w * t_f)
        return target_x, target_y, TARGET_Z, target_vx, target_vy, target_ax, target_ay
        
    else:
        t_land = flight_time - t_fig8_end
        target_z = max(0.0, TARGET_Z - LANDING_SPEED * t_land)
        return home_x, home_y, target_z, 0.0, 0.0, 0.0, 0.0


# =========================================================
# 安全封包發送
# =========================================================
def send_command(sock, ip, port, roll, pitch, yaw_rate, thrust, ts):
    pkt = struct.pack("<4fd", float(roll), float(pitch), float(yaw_rate), float(thrust), float(ts))
    sock.sendto(pkt, (ip, port))


# =========================================================
# 主程式
# =========================================================
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_RECV_PORT))
    sock.setblocking(False)

    print(f"[Ground] 啟動，監聽狀態 port={UDP_RECV_PORT}")
    print(f"[Ground] 傳送命令到 {PI_IP}:{UDP_SEND_PORT}")
    print(f"[Ground] LOG: {LOG_FILE}")

    alt_ctrl = Altitude_ETM_Controller()
    etm_pos_x = Fuzzy_ETM_Core(
        OMEGA_POS, F1_POS, F2_POS, AR_POS,
        "PosX", RATE_LIMIT_XY, GAIN_SCALE_XY,
        POS_DEADZONE_XY, VEL_DEADZONE_XY, SOFT_ERR_XY
    )
    etm_pos_y = Fuzzy_ETM_Core(
        OMEGA_POS, F1_POS, F2_POS, AR_POS,
        "PosY", RATE_LIMIT_XY, GAIN_SCALE_XY,
        POS_DEADZONE_XY, VEL_DEADZONE_XY, SOFT_ERR_XY
    )

    home_initialized = False
    home_x, home_y = 0.0, 0.0

    start_time = 0.0
    prev_time = time.time()
    last_send_time = 0.0
    last_state_time = 0.0
    last_print_time = 0.0

    trig_x_count = 0
    trig_y_count = 0
    trig_z_count = 0
    tx_count = 0

    csv_file = None
    csv_writer = None

    if ENABLE_LOG:
        csv_file = open(LOG_FILE, "w", newline="", encoding="utf-8-sig")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "t", "pi_ts", "clock_diff_ms", "mode",
            "x", "y", "z", "vx", "vy", "vz", "yaw",
            "target_x", "target_y", "target_z", "target_vx", "target_vy",
            "err_w_x", "err_w_y", "err_w_vx", "err_w_vy",
            "err_b_x", "err_b_y", "err_b_vx", "err_b_vy",
            "u_roll", "u_pitch", "u_accel",
            "target_roll", "target_pitch", "yaw_rate_cmd", "thrust_cmd",
            "trig_x", "trig_y", "trig_z", "is_triggered"
        ])

    target_roll = 0.0
    target_pitch = 0.0

    try:
        while True:
            now = time.time()

            if (last_state_time > 0.0) and ((now - last_state_time) > STATE_TIMEOUT):
                try:
                    send_command(sock, PI_IP, UDP_SEND_PORT, 0.0, 0.0, 0.0, 0.0, now)
                except Exception:
                    pass

            try:
                data, addr = sock.recvfrom(1024)

                if len(data) < 36:
                    continue

                x, y, z, vx, vy, vz, yaw, pi_ts = struct.unpack("<7fd", data[:36])

                curr_time = time.time()
                last_state_time = curr_time

                clock_diff_ms = (curr_time - pi_ts) * 1000.0

                vals = [x, y, z, vx, vy, vz, yaw]
                if any(math.isnan(v) or math.isinf(v) for v in vals):
                    continue

                if not home_initialized:
                    home_x = x
                    home_y = y
                    home_initialized = True
                    start_time = curr_time
                    prev_time = curr_time
                    print(f"[Ground] Home 鎖定: x={home_x:.2f}, y={home_y:.2f}")
                    continue

                dt = float(np.clip(curr_time - prev_time, DT_MIN, DT_MAX))
                prev_time = curr_time
                elapsed = curr_time - start_time

                mode_str = "Idle"

                target_x = home_x
                target_y = home_y
                target_z = TARGET_Z
                target_vx = 0.0
                target_vy = 0.0
                target_ax = 0.0
                target_ay = 0.0

                err_w_x = x - target_x
                err_w_y = y - target_y
                err_w_vx = vx - target_vx
                err_w_vy = vy - target_vy

                err_b_x, err_b_y, err_b_vx, err_b_vy = 0.0, 0.0, 0.0, 0.0

                u_roll = 0.0
                u_pitch = 0.0
                u_accel = 0.0

                trig_x = False
                trig_y = False
                trig_z = False
                is_triggered = False

                yaw_rate_cmd = 0.0
                thrust_cmd = 0.0

                if FIXED_CMD_TEST:
                    mode_str = "FixedCmd"
                    target_roll = FIXED_TEST_ROLL
                    target_pitch = FIXED_TEST_PITCH
                    yaw_rate_cmd = FIXED_TEST_YAW_RATE
                    thrust_cmd = FIXED_TEST_THRUST
                    is_triggered = True

                else:
                    if elapsed < SPOOL_TIME:
                        mode_str = "Spool"
                        target_roll = 0.0
                        target_pitch = 0.0
                        thrust_cmd = 0.22
                        is_triggered = True

                    elif elapsed < (SPOOL_TIME + TAKEOFF_TIME):
                        mode_str = "Takeoff"
                        target_roll = 0.0
                        target_pitch = 0.0
                        progress = (elapsed - SPOOL_TIME) / TAKEOFF_TIME
                        thrust_cmd = 0.22 + (HOVER_THRUST - 0.22) * progress
                        is_triggered = True

                    else:
                        flight_time = elapsed - (SPOOL_TIME + TAKEOFF_TIME)
                        mode_str = "Hover" if not ENABLE_FIGURE8 else "Fig-8"

                        target_x, target_y, target_z, target_vx, target_vy, target_ax, target_ay = \
                            generate_trajectory(home_x, home_y, flight_time)

                        err_w_x = x - target_x
                        err_w_y = y - target_y
                        err_w_vx = vx - target_vx
                        err_w_vy = vy - target_vy

                        # ==========================================
                        # 1. XY 軸控制 (優先計算，以提供 Roll/Pitch 給推力補償)
                        # ==========================================
                        if not Z_ONLY:
                            cos_y = math.cos(yaw)
                            sin_y = math.sin(yaw)

                            # 【關鍵修正】對齊 MATLAB：轉為機體座標(Body frame)再送入 ETM
                            err_b_x = err_w_x * cos_y + err_w_y * sin_y
                            err_b_y = -err_w_x * sin_y + err_w_y * cos_y
                            err_b_vx = err_w_vx * cos_y + err_w_vy * sin_y
                            err_b_vy = -err_w_vx * sin_y + err_w_vy * cos_y

                            u_pitch, trig_x, _, _, _, _ = etm_pos_x.update(
                                np.array([err_b_x, err_b_vx], dtype=float), 0.0, dt, curr_time
                            )
                            u_roll, trig_y, _, _, _, _ = etm_pos_y.update(
                                np.array([err_b_y, err_b_vy], dtype=float), 0.0, dt, curr_time
                            )

                            # 前饋控制
                            ff_b_ax = target_ax * cos_y + target_ay * sin_y
                            ff_b_ay = -target_ax * sin_y + target_ay * cos_y
                            ff_pitch_cmd = -ff_b_ax * FF_ACCEL_GAIN
                            ff_roll_cmd = ff_b_ay * FF_ACCEL_GAIN

                            if z < XY_ENABLE_ALTITUDE:
                                target_roll = float(np.clip(-LOW_ALT_XY_GAIN * u_roll, -LOW_ALT_ROLL_PITCH_LIMIT, LOW_ALT_ROLL_PITCH_LIMIT))
                                target_pitch = float(np.clip(LOW_ALT_XY_GAIN * u_pitch, -LOW_ALT_ROLL_PITCH_LIMIT, LOW_ALT_ROLL_PITCH_LIMIT))
                            else:
                                # 【關鍵修正】對齊 MATLAB 正負號
                                target_roll = float(np.clip(-u_roll + ff_roll_cmd, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                                target_pitch = float(np.clip(u_pitch + ff_pitch_cmd, -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT))
                        else:
                            target_roll = 0.0
                            target_pitch = 0.0

                        # ==========================================
                        # 2. Z 軸高度控制 (含推力傾角補償)
                        # ==========================================
                        if not XY_ONLY:
                            current_alt_state = np.array([z, vz], dtype=float)
                            u_accel, trig_z, _, _, _ = alt_ctrl.compute_control(current_alt_state, target_z, dt)
                            
                            # 【關鍵修正】推力傾角補償
                            # 註：由於目前 Python 的 UDP 解包沒有接收真實的 roll/pitch，
                            # 因此這裡使用上一個迴圈產生的目標姿勢 target_roll / target_pitch 來近似推力補償
                            thrust_z_norm = HOVER_THRUST + (u_accel * THRUST_SCALE)
                            tilt_factor = math.cos(target_roll) * math.cos(target_pitch)
                            tilt_factor = max(tilt_factor, 0.5) 
                            
                            thrust_cmd = thrust_z_norm / tilt_factor

                            if z > ALT_SOFT_CEIL:
                                thrust_cmd = min(thrust_cmd, 0.70)
                            if z > ALT_HARD_CEIL:
                                thrust_cmd = min(thrust_cmd, 0.62)

                            thrust_cmd = float(np.clip(thrust_cmd, THR_MIN, THR_MAX))
                        else:
                            thrust_cmd = HOVER_THRUST


                        # ==========================================
                        # 3. ETM 觸發判定
                        # ==========================================
                        time_since_last_send = curr_time - last_send_time
                        # 【關鍵修正】對齊 MATLAB Trig_Flag
                        is_triggered = trig_x or trig_y or trig_z or (time_since_last_send > MAX_SEND_INTERVAL)

                if is_triggered:
                    send_command(sock, PI_IP, UDP_SEND_PORT, target_roll, target_pitch, yaw_rate_cmd, thrust_cmd, curr_time)
                    last_send_time = curr_time
                    tx_count += 1

                if trig_x: trig_x_count += 1
                if trig_y: trig_y_count += 1
                if trig_z: trig_z_count += 1

                if csv_writer is not None:
                    csv_writer.writerow([
                        curr_time, pi_ts, clock_diff_ms, mode_str,
                        x, y, z, vx, vy, vz, yaw,
                        target_x, target_y, target_z, target_vx, target_vy,
                        err_w_x, err_w_y, err_w_vx, err_w_vy,
                        err_b_x, err_b_y, err_b_vx, err_b_vy,
                        u_roll, u_pitch, u_accel,
                        target_roll, target_pitch, yaw_rate_cmd, thrust_cmd,
                        int(trig_x), int(trig_y), int(trig_z), int(is_triggered)
                    ])

                if (curr_time - last_print_time) >= 0.2:
                    last_print_time = curr_time
                    tx_str = "TX" if is_triggered else "--"
                    print(
                        f"[{mode_str}] "
                        f"XYZ=({x:+5.2f},{y:+5.2f},{z:+5.2f}) | "
                        f"Cmd: R={math.degrees(target_roll):+5.1f}°, "
                        f"P={math.degrees(target_pitch):+5.1f}°, "
                        f"Thr={thrust_cmd:.2f} | "
                        f"ClockDiff={clock_diff_ms:7.1f} ms | {tx_str}"
                    )

            except BlockingIOError:
                pass

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\n[Ground] 手動停止")
    finally:
        try:
            send_command(sock, PI_IP, UDP_SEND_PORT, 0.0, 0.0, 0.0, 0.0, time.time())
        except Exception:
            pass

        if csv_file is not None:
            csv_file.close()

        sock.close()

        total_runtime = max(time.time() - start_time, 1e-6) if home_initialized else 0.0
        if total_runtime > 0:
            print(f"[Ground] 總時間: {total_runtime:.2f} s")
            print(f"[Ground] trig_x={trig_x_count}, trig_y={trig_y_count}, trig_z={trig_z_count}, tx={tx_count}")
            print(f"[Ground] tx rate={tx_count / total_runtime:.2f} Hz")
        print("[Ground] 已結束")

if __name__ == "__main__":
    main()