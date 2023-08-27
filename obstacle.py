import numpy as np   
import math
    
class Obstacle:
    def __init__(self, **kwargs):
        self.rows = kwargs.pop('rows',393) # rows 为区域划分的行数, 对应y坐标
        self.cols = kwargs.pop('cols',371) # cols 为列数, 对应x坐标
        self.dx = kwargs.pop('dx',1.21) # dx 为固块长的一半
        self.dy = kwargs.pop('dy',3.95) # dx 为固块宽的一半
        
    
    # 根据障碍物坐标生成obstacles_info
    def obstacle_define(self, obstacle_list): 
        # obstacle_list 存储了障碍物的中心点
        # obstacles_info存储了障碍物的信息, 有障碍物的数组为1, 其他地方为0
        # rows 为区域划分的行数; cols 为列数
        obstacles_info = np.zeros((self.rows, self.cols))
        for i in range(obstacle_list.shape[0]):
            obstacles_info[math.floor(obstacle_list[i, 1]-self.dy ): math.ceil(obstacle_list[i, 1]+self.dy), math.floor(obstacle_list[i, 0]-self.dx) :  math.ceil(obstacle_list[i, 0]+self.dx)] = True
        return obstacles_info
    
    # 定义障碍物信息并进行规避计算
    def obstacle_avoid(self, obstacles_info, X, V, M):
        # BUG 避障速度设为0的话, 后面有分母为0的可能
        # obstacles_info存储了障碍物的信息, 有障碍物的数组为1, 其他地方为0
        # X 为当前粒子的位置
        # V 为修正前的速度
        # V_fixed 为壁障修正后的速度
             
        # 判断粒子的下一步是否在障碍物内部, 并计算距离四个边界的最小距离   
        X_new = X + V
        
        # 做防止X_new超计算域的处理
        if X_new[0, 0] <= M.map_x_lower:
            X_new[0, 0] = M.map_x_lower
            
        if X_new[0, 0] >= M.map_x_upper:
            X_new[0, 0] = M.map_x_upper - 1
            
        if X_new[0, 1] <= M.map_y_lower:
            X_new[0, 1] = M.map_y_lower
            
        if X_new[0, 1] >= M.map_y_upper:
            X_new[0, 1] = M.map_y_upper - 1
        
        # 生成和V相同像形状的数组, 用于存储修正后速度X_new
        V_fixed = np.zeros_like(V)
        
        # 判断粒子的下一位置是否在障碍物内
        if obstacles_info[int(X_new[0, 1]), int(X_new[0, 0])] == 1:
            temp_x_l = X_new[0, 0]
            temp_x_r = X_new[0, 0]
            temp_y_t = X_new[0, 1]
            temp_y_b = X_new[0, 1]
            
            # 当前X坐标距离左右下上边的距离, 分别存储于distance_left, distance_right, distance_bottom, distance_top
            distance_left = 0
            distance_right = 0
            distance_bottom = 0
            distance_top = 0
            
            # 计算距离障碍物左边的距离
            while(obstacles_info[int(X_new[0, 1]), int(temp_x_l)]== 1):
                temp_x_l = temp_x_l - 1
                distance_left = distance_left + 1
                
            while(obstacles_info[int(X_new[0, 1]), int(temp_x_r)] == 1):
                temp_x_r = temp_x_r + 1
                distance_right = distance_right + 1
            
            while(obstacles_info[int(temp_y_b), int(X_new[0, 0])] == 1):
                temp_y_b = temp_y_b - 1
                distance_bottom = distance_bottom + 1  
                
            while(obstacles_info[int(temp_y_t), int(X_new[0, 0])] == 1):
                temp_y_t = temp_y_t + 1
                distance_top = distance_top + 1
    
            distance = np.array([distance_top, distance_bottom, distance_left, distance_right])
            direction_index = np.argmin(distance)           
            min_distance = distance[direction_index]
            
            # 根据障碍物中点距离哪条边最近, 而确定修正法向量的方向, 即N
            if direction_index == 0:
                vector = np.array([0,  1]).reshape(2,1)
            if direction_index == 1:
                vector = np.array([0, -1]).reshape(2,1)
            if direction_index == 2:
                vector = np.array([-1, 0]).reshape(2,1)
            if direction_index == 3:
                vector = np.array([1,  0]).reshape(2,1)
            
            V_fixed = V + (abs(np.dot(V, vector))*vector).reshape(1,2)
             
        else:
            V_fixed = V
                
        return V_fixed 