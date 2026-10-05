import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def generate_cleaning_route_map():
    # 1. 初始化資料：起點與 YOLO 偵測到的髒污點座標 (單位: 公尺)
    # 這些座標對應於您真實場域空拍圖中的物理位置
    home_position = np.array([[0, 0]])
    dirty_spots = np.array([
        [8, 15], [12, 5], [25, 18], [32, 8], 
        [40, 22], [45, 10], [18, 25], [38, 5]
    ])
    
    # 將起點與髒污點合併
    all_points = np.vstack((home_position, dirty_spots))
    num_points = len(all_points)
    
    # 2. 計算距離矩陣 (歐幾里得距離)
    dist_matrix = distance_matrix(all_points, all_points)
    
    # 3. 求解 TSP (採用最近鄰居演算法 Nearest Neighbor)
    unvisited = set(range(1, num_points))
    current_node = 0  # 從 Home 開始
    route = [current_node]
    
    while unvisited:
        # 找出距離當前節點最近的未訪問節點
        next_node = min(unvisited, key=lambda node: dist_matrix[current_node][node])
        route.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
        
    # 4. 將路線座標提取出來準備繪圖
    route_coords = all_points[route]
    
    # 5. 繪製路線地圖
    plt.figure(figsize=(10, 6))
    plt.title('UAV Solar Panel Cleaning - Optimal Route Map', fontsize=14, fontweight='bold')
    
    # 畫出最短路徑連線
    plt.plot(route_coords[:, 0], route_coords[:, 1], linestyle='--', color='blue', alpha=0.7, label='Flight Path')
    
    # 標示髒污點與起點
    plt.scatter(dirty_spots[:, 0], dirty_spots[:, 1], c='red', s=100, zorder=5, label='Dirty Spots (VLA Target)')
    plt.scatter(home_position[:, 0], home_position[:, 1], c='green', marker='*', s=250, zorder=5, label='Home')
    
    # 在圖上標示清洗順序
    for i, node_idx in enumerate(route):
        if i == 0: continue # 略過起點標示
        plt.annotate(f'Stop {i}', (all_points[node_idx][0] + 0.5, all_points[node_idx][1] + 0.5), fontsize=10)

    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # 儲存地圖或直接顯示
    plt.savefig('cleaning_route_map.png', dpi=300)
    plt.show()
    
    return route

# 執行並印出航點順序
optimal_route = generate_cleaning_route_map()
print(f"✅ 最佳航點遍歷順序 (索引值): {optimal_route}")