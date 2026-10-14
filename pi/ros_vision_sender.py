import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import os

class VisionSenderNode(Node):
    def __init__(self):
        super().__init__('vision_sender_node')
        self.bridge = CvBridge()
        # 直接使用 Linux RAM 記憶體共享區，完全繞過網路卡與 Port！
        self.shm_temp = '/dev/shm/uav_frame_tmp.jpg'
        self.shm_final = '/dev/shm/uav_frame.jpg'
        
        topic_name = '/world/solar_farm/model/x500_depth_0/link/camera_link/sensor/IMX214/image'
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, qos_profile)
        
        self.get_logger().info("📡 [Py 3.10] 視覺節點已啟動，正在透過 /dev/shm 記憶體直通車傳送影像...")
        self.frame_count = 0

    def image_callback(self, msg):
        self.frame_count += 1
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_image = cv2.resize(cv_image, (640, 480))
            # 寫入記憶體暫存並透過 os.replace 進行 POSIX 原子更新，保證大腦永遠讀到完整幀
            cv2.imwrite(self.shm_temp, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            os.replace(self.shm_temp, self.shm_final)
            
            if self.frame_count == 1 or self.frame_count % 30 == 0:
                self.get_logger().info(f"✅ 第 {self.frame_count} 幀：已寫入記憶體直通車 ({self.shm_final})")
        except Exception as e:
            self.get_logger().error(f"❌ 影像轉換錯誤: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionSenderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
