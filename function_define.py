# -*- coding: utf-8 -*-
import numpy as np
import re
import pn_class

def taboo_zone_cal(M):
    nrows = M.map_y_upper # rows 为区域划分的行数, 对应y坐标
    ncols = M.map_x_upper # cols 为列数, 对应x坐标
    [x,y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    taboo_zone = np.zeros((nrows, ncols))
    if len(M.taboo_center) != 0: # 
        for i in range(len(M.taboo_center)): # 循环所有禁区
            t = ((x - M.taboo_center[i][0])**2 + (y - M.taboo_center[i][1])**2) < M.taboo_radius**2
            taboo_zone[t] = 1
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.axis ([0, ncols, 0, nrows])
    # ax.imshow(taboo_zone, 'gray')
    return  taboo_zone

# 初始化位置
def position_initial_fixed_position(i): 
    if i == 0:
        return np.array([[335,25]])
    if i == 1:
        return np.array([[335,25]])
    if i == 2:
        return np.array([[335,25]])
    if i == 3:
        return np.array([[335,25]])
    if i == 4:
        return np.array([[335,25]])
    if i == 5:
        return np.array([[335,25]])
    if i == 6:
        return np.array([[335,25]])
    if i == 7:
        return np.array([[335,25]])
    if i == 8:
        return np.array([[335,25]])
    if i == 9:
        return np.array([[335,25]])
    if i == 10:
        return np.array([[335,25]])
    if i == 11:
        return np.array([[335,25]])
    if i == 12:
        return np.array([[335,25]])
    if i == 13:
        return np.array([[335,25]])
    if i == 14:
        return np.array([[335,25]])
    if i == 15:
        return np.array([[335,25]])
    if i == 16:
        return np.array([[335,25]])
    if i == 17:
        return np.array([[335,25]])
    if i == 18:
        return np.array([[335,25]])
    if i == 19:
        return np.array([[335,25]])

def position_initial_fixed_position_around(i): 
    if i == 0:
        return np.array([[107,302]])
    if i == 1:
        return np.array([[155,302]])
    if i == 2:
        return np.array([[214,302]])
    if i == 3:
        return np.array([[260,302]])
    if i == 4:
        return np.array([[90,278]])
    if i == 5:
        return np.array([[90,225]])
    if i == 6:
        return np.array([[90,166]])
    if i == 7:
        return np.array([[90,116]])
    if i == 8:
        return np.array([[107,92]])
    if i == 9:
        return np.array([[155,92]])
    if i == 10:
        return np.array([[214,92]])
    if i == 11:
        return np.array([[335,92]])
    if i == 12:
        return np.array([[280,116]])
    if i == 13:
        return np.array([[280,166]])
    if i == 14:
        return np.array([[280,225]])
    if i == 15:
        return np.array([[280,278]])
    if i == 16:
        return np.array([[335,25]])
    if i == 17:
        return np.array([[335,25]])
    if i == 18:
        return np.array([[335,25]])
    if i == 19:
        return np.array([[335,25]])
    
# 生成随机初始位置, 考虑壁障
# def position_initial_random_position(obstacles_info, dim, map_x_lower, map_x_upper, map_y_lower, map_y_upper):
#     def generate_random_coordinates():
#         x = np.random.uniform(map_x_lower, map_x_upper, size=1)
#         y = np.random.uniform(map_y_lower, map_y_upper, size=1)
#         return x, y
#     X = np.zeros([1, dim])
#     X[0,0], X[0,1] = generate_random_coordinates()
#     ix, iy = np.rint(X[0,0]).astype(int), np.rint(X[0,1]).astype(int)
#     while obstacles_info[iy-1,ix-1] == 1:
#         X[0,0], X[0,1] = generate_random_coordinates()
#         ix, iy = np.rint(X[0,0]).astype(int), np.rint(X[0,1]).astype(int)
#     return X

def position_initial_random_position(dimension, map_x_lower, map_x_upper, map_y_lower, map_y_upper):
    # 初始化各个粒子的位置
    position = np.full((1,dimension), 0)
    position[0,0] = np.random.uniform(map_x_lower+1,  map_x_upper-2) # x坐标(0, x_limit)
    position[0,1] = np.random.uniform(map_y_lower+1,  map_y_upper-2) # y坐标(0, y_limit)
    # 初始化各个粒子的位置, 并进行障碍物的壁障
    # self.position = self.obstacle.obstacle_avoid_init(self.position, self.population, obstacles_info,  self.x_limit, self.y_limit)
    return position

# 速度初始化
def velocity_initial(dim, max_val, min_val):
    velocity = np.random.uniform(min_val,max_val,size=(1, dim))
    return velocity

# 计算小生境的中心
# 通过对所有粒子的位置取均值获得
def niche_center(N, P):
    for i in range(pn_class.niche_num):
        neutral_p = []
        for menber in N[i].menbers:
            b = int(re.findall("\d+",menber)[0])
            neutral_p.append(P[b].position[0])
        neutral_p = np.array(neutral_p)

        xx = np.average(neutral_p[:,0])
        yy = np.average(neutral_p[:,1])
        N[i].center_position = np.array([[xx, yy]])   #(1,2)
    return N

# 利用路径求斥力, i 到 j的距离, i 受 j的力
def repulsion_force_path(P, N, M, niche_radius, robots_number):
            
    # 计算当前小生境位置距另一小生境历史位置的最近距离
    path_dist = np.zeros([pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            min_dist = np.linalg.norm((N[i].center_position - N[j].center_position_history), axis=2)
            path_dist[i, j] = np.linalg.norm(N[i].center_position-N[j].center_position_history[np.argmin(min_dist)])
              
    # 计算小生境与路径间的单位向量, 方向终点减起点
    path_unit_vector = np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            min_dist = np.linalg.norm((N[i].center_position - N[j].center_position_history), axis=2)
            if i == j:
                path_unit_vector[:, i, j] = 0
            else:
                path_unit_vector[:, i, j] = (N[i].center_position - N[j].center_position_history[np.argmin(min_dist)])/np.linalg.norm(N[i].center_position - N[j].center_position_history[np.argmin(min_dist)])
       
    # 计算小生境间的路径斥力, 形状为2*size*size, 因为力是向量
    force_niche_path =  np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])  
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            # 小生境之间的路径斥力 
            if path_dist[i, j] <= niche_radius + niche_radius and i != j:
                force_niche_path[:, i, j] = path_unit_vector[:, i, j] / path_dist[i, j]**2       
   
    # 求小生境所受路径斥力合力
    force_niche_path_summed = np.sum(force_niche_path, axis=2, keepdims=True)         
            
    # 粒子所受合力, 将小生境受力分配到粒子上去
    total_force_particle = np.zeros([robots_number, M.dim])
    for i in range(pn_class.niche_num):
        for menber in N[i].menbers:
            b = int(re.findall("\d+",menber)[0])
            total_force_particle[b, :] =  100*force_niche_path_summed[:, i, :].T
            
    return total_force_particle   

# 更新速度，根据公式V(t+1) = w * V(t) + c1 * r1 * (pbest_i - xi) + c2 * r2 * (gbest_xi - xi)
def velocity_update(V, X, W, force, pbest, gbest, c_ini_1, c_end_1, c_ini_2, c_end_2, w, iter_num, total_iter):
    r1 = np.random.random(1) #随机数
    r2 = np.random.random(1) #随机数
    if np.linalg.norm(pbest - X) != 0 and np.linalg.norm(gbest - X) != 0 and np.linalg.norm(W) != 0:
        alpha_1 = np.arccos(np.dot((pbest - X)[0], (-W)[0])/(np.linalg.norm(pbest - X)*np.linalg.norm(-W)))
        alpha_2 = np.arccos(np.dot((gbest - X)[0], (-W)[0])/(np.linalg.norm(gbest - X)*np.linalg.norm(-W)))
        c1 = 2 * alpha_2/(alpha_1 + alpha_2)
        c2 = 2 * alpha_1/(alpha_1 + alpha_2)
        # c1 = (c_ini_1 + (c_end_1 - c_ini_1)*iter_num/total_iter) * alpha_2/(alpha_1 + alpha_2)
        # c2 = (c_ini_2 + (c_end_2 - c_ini_2)*iter_num/total_iter) * alpha_1/(alpha_1 + alpha_2)
        
    elif np.linalg.norm(pbest - X) == 0 and np.linalg.norm(gbest - X) != 0 and np.linalg.norm(W) != 0:
        alpha_2 = np.arccos(np.dot((gbest - X)[0], (-W)[0])/(np.linalg.norm(gbest - X)*np.linalg.norm(-W)))
        c1 = 0
        c2 = 2
        # c2 = (c_ini_2 + (c_end_2 - c_ini_2)*iter_num/total_iter)
        
    elif np.linalg.norm(pbest - X) != 0 and np.linalg.norm(gbest - X) == 0 and np.linalg.norm(W) != 0:
        alpha_1 = np.arccos(np.dot((pbest - X)[0], (-W)[0])/(np.linalg.norm(pbest - X)*np.linalg.norm(-W)))
        # c1 = (c_ini_1 + (c_end_1 - c_ini_1)*iter_num/total_iter)
        c1 = 2
        c2 = 0
    
    else:
        c1 = 2
        c2 = 2
        
        # c1 = (c_ini_1 + (c_end_1 - c_ini_1)*iter_num/total_iter)
        # c2 = (c_ini_2 + (c_end_2 - c_ini_2)*iter_num/total_iter)
    
    # w = 0.4 + (random.normal(loc=0, scale=1)/10 + random.random()/2)
    w_max = 0.9
    w_min = 0.4
    w = w_max - (w_max - w_min)*iter_num/total_iter
    cognitive = c1*r1*(pbest - X)
    social =  c2*r2*(gbest -X)
    V = w*V + cognitive + social + force
    return V

def velocity_boundary_handle(V, max_val, min_val):       
    V[V < min_val] = min_val
    V[V > max_val] = max_val
    return V

#更新粒子位置，根据公式X(t+1)=X(t)+V, 并约束粒子范围
def position_updata(X, V):
    return X + V
    
def boundary_handle_nm(M, X):
    if X[0][0] < M.map_x_lower:
        X[0][0] = M.map_x_lower
        
    if X[0][1] < M.map_y_lower:
        X[0][1] = M.map_y_lower
        
    if X[0][0] > M.map_x_upper:
        X[0][0] = M.map_x_upper
        
    if X[0][1] > M.map_y_upper:
        X[0][1] = M.map_y_upper        
    
    return X

def boundary_handle(M, X, V):
    if X[0][0] < M.map_x_lower:
        X[0][0] = M.map_x_lower
        
    if X[0][1] < M.map_y_lower:
        X[0][1] = M.map_y_lower
        
    if X[0][0] > M.map_x_upper:
        X[0][0] = M.map_x_upper
        
    if X[0][1] > M.map_y_upper:
        X[0][1] = M.map_y_upper        
    
    return X, -V

def taboo_zone_avoid(X, fitness, taboo_zone):
    if taboo_zone[int(X[0][1])-2,int(X[0][0])-2] == 1:
        return np.array(float(0))
    else:
        return fitness

def check_plume(func, M, X, uds_t):
    # one by one
    fitness = func.fitness_func(X, uds_t)
    if fitness >= M.C_threshold:
        return True
    else:
        return False

# 一些函数的定义                   
def check_boundary(M, X):
    if M.map_x_lower < X[0, 0] < M.map_x_upper and M.map_y_lower < X[0, 1] < M.map_y_upper:
        rebound = False
    else:
        rebound = True
    return rebound

def step_func(X, unit_vector, step):
    return X + unit_vector * step

def turn_45_vector(vector):
    ix = vector[0,0] * np.cos(- np.pi/4) + vector[0,1] *  np.sin(- np.pi/4)
    iy = vector[0,1] *  np.cos(- np.pi/4) - vector[0,0] *  np.sin(- np.pi/4)
    vector =  np.array([[ix, iy]])
    return vector    

def normalize(V):
    norm = np.linalg.norm(V)
    if norm == 0: 
        return V
    else:
        return V / norm
    
def merge_arr_within_distance(arr, distance):
    merged_arr = arr.copy()  # 创建一个原数组的副本，以便进行修改

    i = 0
    while i < len(merged_arr) - 1:
        num1 = merged_arr[i]
        num2 = merged_arr[i + 1]
        if np.sqrt((num1[0]-num2[0])**2 + (num1[1]-num2[1])**2) < distance:
            average = [(num1[0] + num2[0]) / 2,(num1[1] + num2[1]) / 2 ]
            merged_arr[i] = average # 替换第一个数为平均值
            del merged_arr[i + 1]  # 删除第二个数
        else:
            i += 1  # 跳到下一对数进行比较

    return merged_arr

