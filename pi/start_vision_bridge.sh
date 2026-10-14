#!/bin/bash
echo "[Vision Bridge] 啟動影像專用橋接器 (image_bridge)..."
ros2 run ros_gz_image image_bridge /world/solar_farm/model/x500_depth_0/link/camera_link/sensor/IMX214/image
