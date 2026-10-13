#!/bin/bash

# 定義預設的 Gazebo 攝影機 Topic 名稱 (gz_x500_depth 常用預設值)
GZ_TOPIC="/camera/image_raw"
# 定義轉換後的 ROS 2 Topic 資料型態
ROS_TYPE="sensor_msgs/msg/Image"
# 定義原始 Gazebo Topic 資料型態
GZ_TYPE="gz.msgs.Image"

echo "[Vision Bridge] 正在啟動 Gazebo 到 ROS 2 的影像橋接..."
echo "[Vision Bridge] 映射路徑: $GZ_TOPIC -> ROS 2 Topic"

# 執行橋接指令
ros2 run ros_gz_bridge parameter_bridge ${GZ_TOPIC}@${ROS_TYPE}[${GZ_TYPE}