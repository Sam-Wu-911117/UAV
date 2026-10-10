#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
np.float = float

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import socket
import struct
import time
import math
import random
import threading
from collections import deque

# =========================
# [新增] 機載 GAI 與 Web 依賴庫
# =========================
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests

from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleAttitude,
    OffboardControlMode,
    VehicleAttitudeSetpoint,
    VehicleCommand,
    BatteryStatus,
)
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import BatteryState

# =========================
# UDP 設定
# =========================
WINDOWS_IP = "127.0.0.1"
UDP_SEND_PORT = 5006   # Ubuntu -> Windows (state)
UDP_RECV_PORT = 5005   # Windows -> Ubuntu (cmd)

SIM_PACKET_LOSS_RATE = 0.00  
SIM_NETWORK_DELAY = 0.00    
SETPOINT_RATE_HZ = 50.0
CMD_RATE_HZ = 100.0
PRESETPOINT_SECONDS = 1.0
HEARTBEAT_TIMEOUT_S = 0.5
FAILSAFE_THRUST = 0.0
FAILSAFE_LEVEL = True
THR_MIN = 0.10
THR_MAX = 0.90
MAX_TILT = 0.35
DBG_PERIOD_S = 0.2

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

# =======================================================
# [新增] 機載 GAI 接收端與 VLA 大腦初始化 (全域區域)
# =======================================================
app = FastAPI()
gnode_reference = None
vla_policy = None

class UserCommandRequest(BaseModel):
    text_command: str



@app.post("/onboard/command")
def receive_gcs_command(payload: UserCommandRequest):
    global gnode_reference
    user_text = payload.text_command
    
    if gnode_reference is None:
        return {"status": "error", "message": "ROS 2 節點未準備就緒"}
        
    battery = getattr(gnode_reference, 'current_battery_pct', 100.0)
    
    if battery < 20.0:
        return {
            "status": "rejected",
            "reason": f"機載電量過低 ({battery:.1f}%)",
            "mission_plan": [{"action_type": "RTL", "param": 0}]
        }
        
    gnode_reference.get_logger().info(f"呼叫右腦解析: '{user_text}'")
    
    try:
        # [微服務架構] 向本機 5001 Port 的右腦發起推理請求
        res = requests.post("http://127.0.0.1:5001/vla/infer", json={
            "text_command": user_text,
            "battery": battery
        }, timeout=15).json()
        
        if res.get("status") == "success":
            return {
                "status": "success",
                "battery_checked": battery,
                "mission_plan": res.get("mission_plan", [])
            }
        else:
            return {"status": "error", "message": res.get("message")}
            
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "右腦 (VLA微服務) 尚未啟動或連線失敗"}

def run_fastapi_background():
    # 讓 API 運行於 Port 5000
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="warning")
# =======================================================

class PX4DDS_UDP_Offboard_Attitude(Node):
    def __init__(self):
        super().__init__("px4dds_udp_offboard_attitude_bridge")

        # [新增] 電量追蹤變數
        self.current_battery_pct = 100.0

        # ---------- UDP ----------
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", UDP_RECV_PORT))
        self.sock.setblocking(False)
        self.windows_addr = (WINDOWS_IP, UDP_SEND_PORT)

        self.delayed_send_queue = deque()
        self.delayed_recv_queue = deque()

        self.have_lpos = False
        self.have_att = False
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.yaw_enu = 0.0
        self._yaw_ned = 0.0 

        self.cmd_roll = 0.0
        self.cmd_pitch = 0.0
        self.cmd_yaw = 0.0
        self.cmd_thrust = 0.0
        self.last_cmd_time = 0.0
        self.have_cmd = False
        self.echo_time = 0.0  
                      
        self.start_time = time.time()
        self.sent_offboard_cmd = False
        self.sent_arm_cmd = False
        self._last_dbg = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.lpos_cb, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.att_cb, qos)
        self.create_subscription(BatteryStatus, "/fmu/out/battery_status", self.battery_cb, qos)

        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.pub_att_sp = self.create_publisher(VehicleAttitudeSetpoint, "/fmu/in/vehicle_attitude_setpoint_v1", 10)
        self.pub_vehicle_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.pub_battery_state = self.create_publisher(BatteryState, "/drone/battery_state", qos)

        self.create_timer(1.0 / SETPOINT_RATE_HZ, self.offboard_loop)
        self.create_timer(1.0 / CMD_RATE_HZ, self.receive_cmd_from_windows)
        self.create_timer(0.02, self.send_state_to_windows)

        self.get_logger().info("[Bridge PX4DDS Offboard] Started with GAI Onboard Brain.")

    def lpos_cb(self, msg: VehicleLocalPosition):
        self.x = float(msg.y)
        self.y = float(msg.x)
        self.z = -float(msg.z)
        self.vx = float(msg.vy)
        self.vy = float(msg.vx)
        self.vz = -float(msg.vz)
        self.have_lpos = True

    def att_cb(self, msg: VehicleAttitude):
        q = msg.q 
        try:
            _, _, yaw_ned = euler_from_quaternion([float(q[1]), float(q[2]), float(q[3]), float(q[0])])
            self._yaw_ned = wrap_pi(yaw_ned)
            self.yaw_enu = wrap_pi((math.pi / 2.0) - yaw_ned)
            self.have_att = True
        except Exception:
            pass

    def battery_cb(self, msg: BatteryStatus):
        ros_batt_msg = BatteryState()
        ros_batt_msg.voltage = float(msg.voltage_v)
        ros_batt_msg.percentage = float(msg.remaining)
        
        # [新增] 同步更新機載端全域電量
        self.current_battery_pct = float(msg.remaining) * 100.0
        
        ros_batt_msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        ros_batt_msg.present = True
        self.pub_battery_state.publish(ros_batt_msg)

    def send_state_to_windows(self):
        if self.have_lpos and self.have_att:
            if random.random() >= SIM_PACKET_LOSS_RATE:
                try:
                    pkt = struct.pack(
                        "<9fd",
                        self.x, self.y, self.z,
                        self.vx, self.vy, self.vz,
                        self.yaw_enu,
                        self.current_battery_pct,
                        getattr(self, 'current_battery_volt', 25.4),
                        self.echo_time
                    )
                    self.delayed_send_queue.append((time.time() + SIM_NETWORK_DELAY, pkt))
                except Exception:
                    pass

        current_time = time.time()
        while self.delayed_send_queue and current_time >= self.delayed_send_queue[0][0]:
            _, delayed_pkt = self.delayed_send_queue.popleft()
            try:
                self.sock.sendto(delayed_pkt, self.windows_addr)
            except Exception:
                pass

    def receive_cmd_from_windows(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                if random.random() < SIM_PACKET_LOSS_RATE: continue
                self.delayed_recv_queue.append((time.time() + SIM_NETWORK_DELAY, data))
            except socket.error:
                break
            except Exception:
                break

        current_time = time.time()
        while self.delayed_recv_queue and current_time >= self.delayed_recv_queue[0][0]:
            _, data = self.delayed_recv_queue.popleft()
            try:
                if len(data) == 16:
                    r, p, y, t = struct.unpack("<4f", data)
                elif len(data) == 24:
                    r, p, y, t, self.echo_time = struct.unpack("<4f d", data)
                else: continue

                self.cmd_roll = float(max(min(r, MAX_TILT), -MAX_TILT))
                self.cmd_pitch = float(max(min(p, MAX_TILT), -MAX_TILT))
                self.cmd_yaw = float(y)
                self.cmd_thrust = float(max(min(t, THR_MAX), THR_MIN))
                self.last_cmd_time = time.time() 
                self.have_cmd = True
            except Exception:
                pass

    def offboard_loop(self):
        now = time.time()
        self.publish_offboard_control_mode()

        cmd_ok = self.have_cmd and ((now - self.last_cmd_time) <= HEARTBEAT_TIMEOUT_S)
        if cmd_ok:
            roll, pitch, yaw, thrust = self.cmd_roll, self.cmd_pitch, self.cmd_yaw, self.cmd_thrust
        else:
            roll = 0.0 if FAILSAFE_LEVEL else self.cmd_roll
            pitch = 0.0 if FAILSAFE_LEVEL else self.cmd_pitch
            yaw = 0.0
            thrust = FAILSAFE_THRUST

        self.publish_attitude_setpoint(roll, pitch, yaw, thrust)
        
        if (now - self.start_time) >= PRESETPOINT_SECONDS:
            if not self.sent_offboard_cmd:
                self.send_set_mode_offboard()
                self.sent_offboard_cmd = True
            if not self.sent_arm_cmd:
                self.send_arm_command()
                self.sent_arm_cmd = True

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(time.time() * 1e6)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        if hasattr(msg, "actuator"): msg.actuator = False
        self.pub_offboard_mode.publish(msg)

    def publish_attitude_setpoint(self, roll: float, pitch: float, yaw: float, thrust01: float):
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = int(time.time() * 1e6)
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        msg.q_d = [float(qw), float(qx), float(qy), float(qz)]
        msg.thrust_body = [0.0, 0.0, -float(max(min(thrust01, THR_MAX), THR_MIN))]
        self.pub_att_sp.publish(msg)

    def send_arm_command(self):
        self.send_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def send_set_mode_offboard(self):
        self.send_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def send_vehicle_command(self, command: int, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(time.time() * 1e6)
        msg.param1, msg.param2 = float(param1), float(param2)
        msg.command = int(command)
        msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
        msg.from_external = True
        self.pub_vehicle_cmd.publish(msg)

# =======================================================
# 程式進入點
# =======================================================
def main(args=None):
    global gnode_reference
    rclpy.init(args=args)
    node = PX4DDS_UDP_Offboard_Attitude()
    gnode_reference = node  # 將 ROS 節點綁定給 API，讓它能讀取電量
    
       
    # 2. 啟動背景 API 接收地面站文字指令
    api_thread = threading.Thread(target=run_fastapi_background, daemon=True)
    api_thread.start()
    node.get_logger().info("🚀 機載端 GAI 接收服務已在 Port 5000 背景啟動。")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()