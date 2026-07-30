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
from collections import deque

from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleAttitude,
    OffboardControlMode,
    VehicleAttitudeSetpoint,
    VehicleCommand,
)
from tf_transformations import euler_from_quaternion, quaternion_from_euler


# =========================
# UDP 設定
# =========================
WINDOWS_IP = "10.203.103.16"
UDP_SEND_PORT = 5006   # Ubuntu -> Windows (state)
UDP_RECV_PORT = 5005   # Windows -> Ubuntu (cmd)

# =========================
# 網路模擬設定 (Network Simulation)
# =========================
SIM_PACKET_LOSS_RATE = 0.00  # 掉包率 (0.3 代表 30%)
SIM_NETWORK_DELAY = 0.00    # 延遲時間 (秒)

# =========================
# Offboard 參數
# =========================
SETPOINT_RATE_HZ = 50.0
CMD_RATE_HZ = 100.0

PRESETPOINT_SECONDS = 1.0
HEARTBEAT_TIMEOUT_S = 0.5

FAILSAFE_THRUST = 0.0
FAILSAFE_LEVEL = True

THR_MIN = 0.10
THR_MAX = 0.90

MAX_TILT = 0.35

# debug print period
DBG_PERIOD_S = 0.2


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PX4DDS_UDP_Offboard_Attitude(Node):
    def __init__(self):
        super().__init__("px4dds_udp_offboard_attitude_bridge")

        # ---------- UDP ----------
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", UDP_RECV_PORT))
        self.sock.setblocking(False)
        self.windows_addr = (WINDOWS_IP, UDP_SEND_PORT)

        # ---------- 網路模擬佇列 ----------
        self.delayed_send_queue = deque()
        self.delayed_recv_queue = deque()

        # ---------- State ----------
        self.have_lpos = False
        self.have_att = False
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0

        self.yaw_enu = 0.0
        self._yaw_ned = 0.0  # debug

        # ---------- Last cmd ----------
        self.cmd_roll = 0.0
        self.cmd_pitch = 0.0
        self.cmd_yaw = 0.0
        self.cmd_thrust = 0.0
        self.last_cmd_time = 0.0
        self.have_cmd = False
        self.echo_time = 0.0  # [新增] 用來儲存 Windows 傳過來的時間戳記

        # ---------- Offboard ----------
        self.start_time = time.time()
        self.sent_offboard_cmd = False
        self.sent_arm_cmd = False

        # ---------- Debug ----------
        self._last_dbg = 0.0

        # ---------- QoS ----------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------- Subscriptions ----------
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.lpos_cb,
            qos,
        )
        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.att_cb,
            qos,
        )

        # ---------- Publishers ----------
        self.pub_offboard_mode = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.pub_att_sp = self.create_publisher(
            VehicleAttitudeSetpoint, "/fmu/in/vehicle_attitude_setpoint_v1", 10
        )
        self.pub_vehicle_cmd = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )

        # ---------- Timers ----------
        self.create_timer(1.0 / SETPOINT_RATE_HZ, self.offboard_loop)
        self.create_timer(1.0 / CMD_RATE_HZ, self.receive_cmd_from_windows)
        self.create_timer(0.02, self.send_state_to_windows)

        self.get_logger().info("[Bridge PX4DDS Offboard Attitude] Started with Network Simulation.")
        self.get_logger().info(f"Loss Rate: {SIM_PACKET_LOSS_RATE*100}%, Delay: {SIM_NETWORK_DELAY*1000} ms")
        self.get_logger().info(f"Windows IP: {WINDOWS_IP}")
        self.get_logger().info(f"UDP listen(cmd): {UDP_RECV_PORT}, UDP send(state): {UDP_SEND_PORT}")

    # =========================
    # PX4 DDS callbacks
    # =========================
    def lpos_cb(self, msg: VehicleLocalPosition):
        x_n, y_e, z_d = float(msg.x), float(msg.y), float(msg.z)
        vx_n, vy_e, vz_d = float(msg.vx), float(msg.vy), float(msg.vz)

        self.x = y_e
        self.y = x_n
        self.z = -z_d
        self.vx = vy_e
        self.vy = vx_n
        self.vz = -vz_d

        self.have_lpos = True

    def att_cb(self, msg: VehicleAttitude):
        q = msg.q  # [w, x, y, z]
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        try:
            _, _, yaw_ned = euler_from_quaternion([qx, qy, qz, qw])
            self._yaw_ned = wrap_pi(yaw_ned)
            self.yaw_enu = wrap_pi((math.pi / 2.0) - yaw_ned)
            self.have_att = True
        except Exception:
            pass

    # =========================
    # UDP: state -> Windows (加入延遲與掉包)
    # =========================
    def send_state_to_windows(self):
        if self.have_lpos and self.have_att:
            if random.random() >= SIM_PACKET_LOSS_RATE:
                try:
                    # [修改] 直接打包 echo_time 回傳給 Windows
                    pkt = struct.pack(
                        "<7fd",
                        self.x, self.y, self.z,
                        self.vx, self.vy, self.vz,
                        self.yaw_enu,
                        self.echo_time,
                    )
                    execute_time = time.time() + SIM_NETWORK_DELAY
                    self.delayed_send_queue.append((execute_time, pkt))
                except Exception:
                    pass

        current_time = time.time()
        while self.delayed_send_queue and current_time >= self.delayed_send_queue[0][0]:
            _, delayed_pkt = self.delayed_send_queue.popleft()
            try:
                self.sock.sendto(delayed_pkt, self.windows_addr)
            except Exception:
                pass

    # =========================
    # UDP: cmd <- Windows (加入延遲與掉包)
    # =========================
    def receive_cmd_from_windows(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                
                if random.random() < SIM_PACKET_LOSS_RATE:
                    continue

                process_time = time.time() + SIM_NETWORK_DELAY
                self.delayed_recv_queue.append((process_time, data))
                
            except socket.error:
                break
            except Exception as e:
                self.get_logger().error(f"UDP recv error: {e}")
                break

        current_time = time.time()
        while self.delayed_recv_queue and current_time >= self.delayed_recv_queue[0][0]:
            _, data = self.delayed_recv_queue.popleft()
            try:
                if len(data) == 16:
                    r, p, y, t = struct.unpack("<4f", data)
                elif len(data) == 24:
                    # [修改] 將收到的 Windows 時間戳記存入 self.echo_time
                    r, p, y, t, self.echo_time = struct.unpack("<4f d", data)
                else:
                    continue

                r = float(max(min(r, MAX_TILT), -MAX_TILT))
                p = float(max(min(p, MAX_TILT), -MAX_TILT))
                y = float(y)
                t = float(max(min(t, THR_MAX), THR_MIN))

                self.cmd_roll = r
                self.cmd_pitch = p
                self.cmd_yaw = y
                self.cmd_thrust = t
                self.last_cmd_time = time.time() 
                self.have_cmd = True

            except Exception as e:
                pass

    # =========================
    # PX4 Offboard loop
    # =========================
    def offboard_loop(self):
        now = time.time()

        self.publish_offboard_control_mode()

        cmd_ok = self.have_cmd and ((now - self.last_cmd_time) <= HEARTBEAT_TIMEOUT_S)

        if cmd_ok:
            roll = self.cmd_roll
            pitch = self.cmd_pitch
            yaw = self.cmd_yaw
            thrust = self.cmd_thrust
        else:
            roll = 0.0 if FAILSAFE_LEVEL else self.cmd_roll
            pitch = 0.0 if FAILSAFE_LEVEL else self.cmd_pitch
            yaw = 0.0
            thrust = FAILSAFE_THRUST

        self.publish_attitude_setpoint(roll, pitch, yaw, thrust)

        if (now - self._last_dbg) >= DBG_PERIOD_S:
            self._last_dbg = now
            self.get_logger().info(
                f"ENU x={self.x:+.2f} y={self.y:+.2f} z={self.z:+.2f} | "
                f"v={self.vx:+.2f},{self.vy:+.2f},{self.vz:+.2f} | "
                f"yaw_enu={self.yaw_enu:+.2f} | "
                f"cmd r={roll:+.2f} p={pitch:+.2f} y={yaw:+.2f} thr={thrust:+.2f} | "
                f"cmd_ok={cmd_ok}"
            )

        if (now - self.start_time) >= PRESETPOINT_SECONDS:
            if not self.sent_offboard_cmd:
                self.send_set_mode_offboard()
                self.sent_offboard_cmd = True
                self.get_logger().info("Sent OFFBOARD mode command.")

            if not self.sent_arm_cmd:
                self.send_arm_command()
                self.sent_arm_cmd = True
                self.get_logger().info("Sent ARM command.")

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(time.time() * 1e6)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        if hasattr(msg, "actuator"):
            msg.actuator = False
        self.pub_offboard_mode.publish(msg)

    def publish_attitude_setpoint(self, roll: float, pitch: float, yaw: float, thrust01: float):
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = int(time.time() * 1e6)
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        msg.q_d = [float(qw), float(qx), float(qy), float(qz)]
        thr = float(max(min(thrust01, THR_MAX), THR_MIN))
        msg.thrust_body = [0.0, 0.0, -thr]
        self.pub_att_sp.publish(msg)

    def send_arm_command(self):
        self.send_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def send_set_mode_offboard(self):
        self.send_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def send_vehicle_command(self, command: int, param1=0.0, param2=0.0, param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(time.time() * 1e6)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.pub_vehicle_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PX4DDS_UDP_Offboard_Attitude()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
