import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import os
import json

class OnDemandVisionSender(Node):
    def __init__(self):
        super().__init__('on_demand_vision_sender')
        self.bridge = CvBridge()
        
        self.shm_req_path = '/dev/shm/uav_req.json'
        self.shm_img_temp = '/dev/shm/uav_frame_tmp.jpg'
        self.shm_img_final = '/dev/shm/uav_frame.jpg'
        self.shm_meta_temp = '/dev/shm/uav_frame_meta_tmp.json'
        self.shm_meta_final = '/dev/shm/uav_frame_meta.json'
        
        self.latest_msg = None
        self.last_handled_req_id = -1
        
        topic_name = '/world/solar_farm/model/x500_depth_0/link/camera_link/sensor/IMX214/image'
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, qos_profile)
        
        # 每 0.05 秒 (20Hz) 輕量檢查一次地面站是否抵達髒污點並請求影像
        self.req_timer = self.create_timer(0.05, self.check_request_callback)
        
        self.get_logger().info("📡 [Py 3.10 按需視覺節點] 已啟動！平時零負載待命，抵達髒污點才發送影像...")

    def image_callback(self, msg):
        # 平時只保存最新一幀的指標，不做任何影像編碼與硬碟寫入，保證相機完全不卡頓！
        self.latest_msg = msg

    def check_request_callback(self):
        if self.latest_msg is None or not os.path.exists(self.shm_req_path):
            return
        try:
            with open(self.shm_req_path, 'r') as f:
                req = json.load(f)
            req_id = req.get("req_id", -1)
            wp_idx = req.get("wp_index", 0)
            
            # 只有當收到全新的請求編號時，才抓拍 1 張影像送給 VLA
            if req_id > self.last_handled_req_id:
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding="bgr8")
                cv_image = cv2.resize(cv_image, (640, 480))
                
                # 1. 原子寫入單張 JPEG 影像
                cv2.imwrite(self.shm_img_temp, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                os.replace(self.shm_img_temp, self.shm_img_final)
                
                # 2. 原子寫入影像對應的 frame_id 標籤，通知 VLA 影像已就緒
                with open(self.shm_meta_temp, 'w') as f:
                    json.dump({"frame_id": req_id, "wp_index": wp_idx}, f)
                os.replace(self.shm_meta_temp, self.shm_meta_final)
                
                self.last_handled_req_id = req_id
                self.get_logger().info(f"📸 [抵達髒污點 #{wp_idx}] 已應地面站請求 (req_id={req_id}) 抓拍並發送 1 幀高清影像予 SmolVLA！")
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = OnDemandVisionSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
