#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import os

class ContinuousTreeVisionSender(Node):
    def __init__(self):
        super().__init__('continuous_tree_vision_sender')
        self.bridge = CvBridge()
        
        # 共享記憶體路徑 (簡化為僅提供連續影像)
        self.shm_img_temp = '/dev/shm/uav_frame_tmp.jpg'
        self.shm_img_final = '/dev/shm/uav_frame.jpg'
        
        self.latest_msg = None
        self.frame_count = 0
        
        # 宣告相機 Topic 參數
        self.declare_parameter(
            'camera_topic',
            '/world/tree_farm/model/x500_depth_0/link/camera_link/sensor/IMX214/image'
        )
        topic_name = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, qos_profile)
        
        # 改為連續截圖計時器 (例如 10 Hz = 0.1秒，配合 VLA 一般推論速度)
        self.stream_timer = self.create_timer(0.1, self.stream_callback)
        
        self.get_logger().info(f"📡 [連續視覺串流節點] 已啟動！訂閱 Topic: {topic_name}")
        self.get_logger().info("🌲 配合無雲台動態追蹤，現已改為 10Hz 連續寫入共享記憶體，供 VLA 即時閉環推論。")

    def image_callback(self, msg):
        self.latest_msg = msg

    def stream_callback(self):
        if self.latest_msg is None:
            return
            
        try:
            # 將 ROS 2 Image 轉換為 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding="bgr8")
            cv_image = cv2.resize(cv_image, (640, 480))
            
            # 原子寫入 JPEG 影像至共享記憶體，避免推論端讀到不完整的破圖
            cv2.imwrite(self.shm_img_temp, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            os.replace(self.shm_img_temp, self.shm_img_final)
            
            self.frame_count += 1
            if self.frame_count % 50 == 0:
                self.get_logger().info(f"📸 持續提供影像串流中... (已刷新 {self.frame_count} 幀)")
                
        except Exception as e:
            self.get_logger().error(f"❌ [錯誤] 處理連續截圖時發生異常: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ContinuousTreeVisionSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()