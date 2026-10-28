#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import os
import json
import time

class OnDemandTreeVisionSender(Node):
    def __init__(self):
        super().__init__('on_demand_tree_vision_sender')
        self.bridge = CvBridge()
        
        # 共享記憶體路徑 (與地面站及推論端對齊)
        self.shm_req_path = '/dev/shm/uav_req.json'
        self.shm_img_temp = '/dev/shm/uav_frame_tmp.jpg'
        self.shm_img_final = '/dev/shm/uav_frame.jpg'
        self.shm_meta_temp = '/dev/shm/uav_frame_meta_tmp.json'
        self.shm_meta_final = '/dev/shm/uav_frame_meta.json'
        
        self.latest_msg = None
        self.last_handled_req_id = -1
        
        # 宣告相機 Topic 參數 (預設為樹林場景或通用相機 Topic，可由 ros2 run --ros-args -p 覆寫)
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
        
        # 20 Hz 定時檢查地面站發來的入弧前哨截圖請求
        self.req_timer = self.create_timer(0.05, self.check_request_callback)
        
        self.get_logger().info(f"📡 [樹列巡檢按需視覺節點] 已啟動！訂閱 Topic: {topic_name}")
        self.get_logger().info("🌲 平時輕量待命，無人機抵達單側外擺圓弧切入點時觸發截幀推論...")

    def image_callback(self, msg):
        self.latest_msg = msg

    def check_request_callback(self):
        if self.latest_msg is None or not os.path.exists(self.shm_req_path):
            return
        try:
            with open(self.shm_req_path, 'r') as f:
                req = json.load(f)
            req_id = req.get("req_id", -1)
            tree_idx = req.get("tree_index", req.get("wp_index", 0))
            task_name = req.get("task", "ALIGN_TREE_SINGLE_SIDE")
            
            if req_id > self.last_handled_req_id:
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding="bgr8")
                cv_image = cv2.resize(cv_image, (640, 480))
                
                # 1. 原子寫入單張 JPEG 影像至共享記憶體
                cv2.imwrite(self.shm_img_temp, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                os.replace(self.shm_img_temp, self.shm_img_final)
                
                # 2. 原子寫入中繼資料
                meta_data = {
                    "req_id": req_id,
                    "tree_index": tree_idx,
                    "task": task_name,
                    "timestamp": time.time()
                }
                with open(self.shm_meta_temp, 'w') as f:
                    json.dump(meta_data, f)
                os.replace(self.shm_meta_temp, self.shm_meta_final)
                
                self.last_handled_req_id = req_id
                self.get_logger().info(f"📸 [抵達樹木 #{tree_idx} 外擺切入點] 成功應請求 (req_id={req_id}) 抓拍並發送 1 幀影像至推論引擎！")
        except Exception as e:
            self.get_logger().error(f"❌ [錯誤] 處理截圖請求時發生異常: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OnDemandTreeVisionSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()