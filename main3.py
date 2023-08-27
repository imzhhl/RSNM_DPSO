# -*- coding: utf-8 -*-
"""
重构一下
"""
import numpy as np
import re
from fluent2python import Fluent2python
from fitness_wind_function import Fitness, Wind
from obstacle import Obstacle
from figure_plot import Figure_plot
from pn_class import Particle, Niche, Map
import pn_class
from function_define import taboo_zone_cal, position_initial_fixed_position, \
                            velocity_initial, niche_center, repulsion_force_path,\
                            velocity_update,velocity_boundary_handle, position_updata,\
                            boundary_handle, taboo_zone_avoid, position_initial_random_position,\
                            check_plume, check_boundary, step_func, turn_45_vector,normalize,\
                            merge_arr_within_distance

def P_menber(menbers):
    m = []
    for i in range(len(menbers)):
        m.append(int(re.findall("\d+", menbers[i])[0]))
    return m

def nelder_mead(P, N, M, uds_t, func, no_improve_thr=10e-6, no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    
    m = P_menber(N.menbers)
    # init
    # 更新所有粒子的适应值
    for i in m:
        P[i].fitness = func.fitness_func(P[i].position, uds_t)
    # 降序排列,只需要将N.menber里面的内容排序即可
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if P[m[i]].fitness < P[m[j]].fitness:
                N.menbers[i], N.menbers[j] = N.menbers[j], N.menbers[i]
    
    m = P_menber(N.menbers)
    
    # order
    no_improve_thr=0.01
    no_improv_break=100
    best = P[m[0]].fitness 

    # break after no_improv_break iterations with no improvement
    # print ('...best so far:', best)
    prev_best = 0
    no_improv = 0

    if best > prev_best + no_improve_thr:        #如果最大值比之前的最大值要大
        # no_improv = 0
        prev_best = best
    else:                                        # 否则
        no_improv = no_improv + 1                # 没有更新prev_best的次数+1
        # print(self.no_improv)

    if no_improv >= no_improv_break:             # 如果超过10次未更新
        return P,N

    # centroid
    x0 = 0                                       # 初始化前(n-1)项平均值
    for i in range(len(m[0:-1])):                # 剔除res中最后一行，进行遍历
        x0 = x0 + P[m[i]].position[0]/len(m[0:-1])


    # reflection
    xr = x0 + alpha*(x0 - P[m[-1]].position)     # alpha=1.  求最小值点关于其他点平均值点的对称点
    boundary_handle(M, xr)                       # 防超域处理
    rscore = func.fitness_func(xr, uds_t)  
    
      
    if P[m[-2]].fitness <= rscore < P[m[0]].fitness:# 如果对称点的值不大不小
        # P[-1].position = NM_obstacle_avoid(P[-1].position, xr)   # 直接将对称点替换掉最小的点
        P[m[-1]].position = xr
        P[m[-1]].fitness = func.fitness_func(P[m[-1]].position, uds_t)
        return P,N                               # 后面不运行了，进入下一次循环

    # expansion
    elif rscore > P[m[0]].fitness:               # 如果对称点的值最大（说明这个方向是对的，所以多加点）
        xe = x0 + gamma*(x0 - P[m[-1]].position) # gamma=2.   求最小值点关于其他点平均值点更远的对称点
        boundary_handle(M, xe)                   # 防超域处理
        escore = func.fitness_func(xe, uds_t)
        
        if escore > rscore:                      # 如果更远的对称点的值更大
            # P[-1].position = NM_obstacle_avoid(P[-1].position, xe)               # 直接将对称点替换掉最小的点
            P[m[-1]].position = xe
            P[m[-1]].fitness = func.fitness_func(P[m[-1]].position, uds_t)
            return P,N
        else:
            # P[-1].position = NM_obstacle_avoid(P[-1].position, xr)   # 直接将对称点替换掉最小的点
            P[m[-1]].position = xr
            P[m[-1]].fitness = func.fitness_func(P[m[-1]].position, uds_t)
            return P,N

    # contraction                                # 如果对称点的值比本来的最小值都还要小（说明方向不对，要缩短一点）
    elif rscore <= P[m[-1]].fitness:
        xc = x0 + rho*(x0 - P[m[-1]].position)   # rho=-0.5    求最小值点关于其他点平均值点更近一点的对称点
        boundary_handle(M, xc)                   # 防超域处理
        cscore = func.fitness_func(xc, uds_t)
        if cscore > P[m[-1]].fitness:            # 如果更近的对称点的值比最小值要大
            # P[-1].position = NM_obstacle_avoid(P[-1].position, xc)               # 直接将比对称点更近的点替换掉最小的点
            P[m[-1]].position = xc
            P[m[-1]].fitness = func.fitness_func(P[m[-1]].position, uds_t)         # 把这个点加进去
            return P,N

        # reduction                           
        if cscore <= P[m[-1]].fitness:           # 如果更近的对称点的值比最小值还要小
            x1 = P[m[0]].fitness                 # x1为最大值
            
            for tup in P:
                tup.position = x1 + sigma*(tup.position - x1)         # sigma=0.5    每个点关于最大值点的近一点的对称点
                boundary_handle(M, tup.position)                      # 防超域处理
                tup.fitness = func.fitness_func(tup.position, uds_t)  # 将之前所有点都关于最小值点对称过去，重新组成新的点
    
    N.gbest_fitness = P[m[0]].fitness
    N.gbest_position = P[m[0]].position

                            
def find_func():
    M = Map()                       # 实例化地图
    list_old = M.taboo_center.copy() # 判断列表是否发生变化
    taboo_zone = taboo_zone_cal(M)
    func = Fitness()                # 实例化适应度类
    wind = Wind()                   # 实例化风向类
    figure = Figure_plot()          # 实例化绘图类
    obstacle = Obstacle()           # 实例化障碍物类
    obstacles_info = obstacle.obstacle_define(M.obstacle_list) # 获取障碍物数组
    fluent_python = Fluent2python() # 实例化fluent数据类
    # uds_t, grid_x, grid_y = fluent_python.file_to_array()      # 导入fluent数据
    uds_t = np.load("uds_t.npy")
    grid_x = np.load("grid_x.npy")
    grid_y = np.load("grid_y.npy")
    
    # 小生境实例化
    N = []
    for i in range(pn_class.niche_num):
        N.append(Niche())
        N[i].NO = i
        
    # 粒子实例化
    P = []
    for i in range(pn_class.particle_num):
        P.append(Particle())
        P[i].NO = i
        # 小生境划分,前一半属于小生境0,后一半属于小生境1
        if 0<= i < pn_class.particle_num/2 :
            P[i].belonged_niche_NO = 0
        else:
            P[i].belonged_niche_NO = 1
            
        P[i].position = position_initial_fixed_position(i) # 固定位置初始化
        # P[i].position = position_initial_random_position(obstacles_info, M.dim, M.map_x_lower, M.map_x_upper, M.map_y_lower, M.map_y_upper) # 随机位置初始化
        P[i].velocity = velocity_initial(M.dim, P[i].max_val, P[i].min_val) # 速度初始化
        P[i].unit_vector = normalize(P[i].velocity)
        P[i].fitness = func.fitness_func(P[i].position, uds_t)
    
        # 记录
        P[i].fitness_history.append(P[i].fitness)
        P[i].position_history.append(P[i].position)    
    
    # 初始化
    # 更新粒子信息
    for i in range(pn_class.particle_num):
        P[i].fitness = func.fitness_func(P[i].position, uds_t)
        P[i].pbest_fitness = P[i].fitness    # 初始化
        P[i].pbest_position = P[i].position  # 初始化
      
    # 初始化全局最优值
    for i in range(pn_class.niche_num):
        N[i].menbers = []
        N[i].UAVs_position = dict()
        N[i].UAVs_fitness = dict() 
        for j in range(pn_class.particle_num):
            if P[j].belonged_niche_NO == N[i].NO:
                N[i].menbers.append('UAVs_'+str(j))
                N[i].UAVs_position['UAVs_'+str(j)] = P[j].position
                N[i].UAVs_fitness['UAVs_'+str(j)]  = P[j].fitness
        
        i_gbest = max(N[i].UAVs_fitness, key=lambda x:N[i].UAVs_fitness[x])
        N[i].gbest_fitness = N[i].UAVs_fitness[i_gbest]
        N[i].gbest_position = N[i].UAVs_position[i_gbest]  
        # 记录
        N[i].gbest_position_history.append(N[i].gbest_position)
        N[i].gbest_fitness_history.append(N[i].gbest_fitness)
    
# %% 开始循环
    for iter_num in range(pn_class.total_iter):
        #---------------检测列表是否反生变化
        list_new = M.taboo_center # 判断列表是否发生变化
        if list_new != list_old:
            taboo_zone = taboo_zone_cal(M)
            list_old = list_new.copy()
        #--------------- 
        # =========================================================================
        # 第一阶段, 羽流发现         
        # =========================================================================  
        for i in range(pn_class.particle_num):
            # 计算每个无人机适应度
            P[i].fitness = func.fitness_func(P[i].position, uds_t)
            # 将新适应度增加到列表中
            P[i].fitness_history.append(P[i].fitness)
            # 羽流检测
            P[i].in_plume = check_plume(func, M, P[i].position, uds_t)    
            # 碰壁检测
            P[i].rebound = check_boundary(M, P[i].position)
            # 如果rebound为false, 则执行step_func前进一步, 在羽流中时则不再进行此更新
            if P[i].rebound == False and P[i].in_plume == False:
                # 记录历史移动位置
                P[i].position = step_func(P[i].position, P[i].unit_vector, P[i].step)
                # 限制粒子位置
                P[i].position = boundary_handle(M, P[i].position) 
                P[i].position_history.append(P[i].position.tolist()[0])
                # 躲避禁区
                P[i].position, P[i].velocity = taboo_zone_avoid(P[i].position, P[i].velocity, M, taboo_zone)
                # 限制粒子位置
                P[i].position = boundary_handle(M, P[i].position) 
    
                # 如果前进后碰壁, 则回退一步, 并进行20米的反弹
                if P[i].rebound == True:
                    P[i].unit_vector = -P[i].unit_vector
                    P[i].position = step_func(P[i].position, P[i].unit_vector, 10*P[i].step)
                    # 限制粒子位置
                    P[i].position = boundary_handle(M, P[i].position) 
                    P[i].position_history.append(P[i].position.tolist()[0])
                    # 再随机一个初始方向
                    # V = random.uniform(size=(1,robots.UAVs.dimension))
                    # U[i].unit_vector = normalize(V)
                    
                    # 再旋转45度弹出
                    P[i].unit_vector = turn_45_vector(P[i].unit_vector) 
                    
            # 循环所有的无人机
            for i in range(pn_class.particle_num):  
                if P[i].in_plume == True:
                    N[P[i].belonged_niche_NO].in_PSO = True
            

        # =========================================================================
        # 第二阶段, 羽流追踪        
        # =========================================================================             
        # 接入PSO, 每个niche独立进行PSO         
        
        # 计算小生境的中心
        N = niche_center(N, P)
        for i in range(pn_class.niche_num):
            N[i].center_position_history.append(N[i].center_position)
    
        # 每个粒子所受的力
        total_force = repulsion_force_path(P, N, M) # 利用路径方法求斥力
        for i in range(pn_class.particle_num):
            P[i].force = total_force[i].reshape(1,2)
        
        # 更新粒子的速度, one by one for particles
        for i in range(pn_class.niche_num):
            # 计算小生境的聚集度 
            dmin = 10
            dmax = 100    
            s_sum = 0 # 初始化相似度
            for menber in N[i].menbers:
                d = np.linalg.norm(N[i].UAVs_position[menber] - N[i].gbest_position)
                if d <= dmin:
                    s = 1
                elif dmin < d < dmax:
                    s = 1 - d / N[i].niche_Gr
                elif d >= dmax:
                    s = 0
                s_sum = s_sum + s
            N[i].agregation = s_sum / len(N[i].menbers)
         
            # 如果聚集度大于0.3则开始进行Nelder-Mead算法
            if N[i].in_PSO == True and N[i].in_NM == False:
                if  N[i].agregation > N[i].epsilon_nm and N[i].gbest_fitness > N[i].C_threshold_nm:   #这里如果更换数据，相应阈值也要变化
                    N[i].in_NM = True
            
            # 开始PSO算法
            if N[i].in_PSO == True and N[i].in_NM == False:
                
                # 更新所有小生境中各个字典的信息
                for menber in N[i].menbers:
                    b = int(re.findall("\d+",menber)[0])
                    N[i].UAVs_position['UAVs_' + str(b)] = P[b].position       # UAVs_position应该是每步变化的
                    N[i].UAVs_fitness['UAVs_' + str(b)] = P[b].fitness         # UAVs_fitness应该是每步变化的

                for menber in N[i].menbers:
                    b = int(re.findall("\d+",menber)[0])
                    P[b].velocity = velocity_update(P[b].velocity,
                                                    P[b].position, 
                                                    P[b].wind, 
                                                    P[b].force, 
                                                    P[b].pbest_position, 
                                                    N[i].gbest_position,
                                                    P[b].c1, 
                                                    P[b].c2,
                                                    P[b].w)
                    P[b].velocity = velocity_boundary_handle(P[b].velocity, P[b].max_val, P[b].min_val)
                    # P[b].velocity = obstacle.obstacle_avoid(obstacles_info, P[b].position, P[b].velocity)
                    P[b].position = position_updata(P[b].position, P[b].velocity, N[i].restriction_radius, N[i].center_position)
                    P[b].position = boundary_handle(M, P[b].position)
                    P[b].position, P[b].velocity = taboo_zone_avoid(P[b].position, P[b].velocity, M, taboo_zone)
                    P[b].position = boundary_handle(M, P[b].position)
                    P[b].fitness = func.fitness_func(P[b].position, uds_t)
                    P[b].fitness2 = func.fitness_func(P[b].position, uds_t)
                    
                    # 更新pbest_fitness
                    if P[b].fitness2 >= P[b].pbest_fitness:
                        P[b].pbest_position = P[b].position
                        P[b].pbest_fitness = P[b].fitness2
                    
                    # 更新gbest_fitness
                    if P[b].fitness2 >= N[i].gbest_fitness:
                        N[i].gbest_fitness = P[b].fitness2
                        N[i].gbest_position = P[b].position
                        
            N[i].gbest_position_history.append(N[i].gbest_position)
            N[i].gbest_fitness_history.append(N[i].gbest_fitness)
            
            # 开始NM算法
            if N[i].in_PSO == True and N[i].in_NM == True:
                nelder_mead(P, N[i], M, uds_t, func)
                # 记录
                N[i].gbest_position_history.append(N[i].gbest_position)
                N[i].gbest_fitness_history.append(N[i].gbest_fitness)
            
        for i in range(pn_class.particle_num):
            # 记录
            P[i].fitness_history.append(P[i].fitness)
            P[i].position_history.append(P[i].position)
            
            
            
        # 设置搜索禁忌区
        for i in range(pn_class.niche_num):
            # 三个判断条件
            if  N[i].agregation > N[i].epsilon_end and N[i].gbest_fitness > N[i].C_threshold_end:
                M.taboo_center.append([N[i].gbest_position[0][0],N[i].gbest_position[0][1]])
                # 合并距离较近的两个源
                M.taboo_center = merge_arr_within_distance(M.taboo_center, 10)
                
                
                M.taboo_center.append([N[i].gbest_position[0][0],N[i].gbest_position[0][1]])
                # 合并距离较近的两个源
                M.taboo_center = merge_arr_within_distance(M.taboo_center, 10)
                
                for menber in N[i].menbers:
                    P[b].fitness = float('-inf')
                    P[b].pbest_fitness = P[b].fitness
                    P[b].position = position_initial_random_position(obstacles_info, M.dim, M.map_x_lower, M.map_x_upper, M.map_y_lower, M.map_y_upper)
                    P[b].pbest_position = P[b].position
                    P[b].velocity = velocity_initial(M.dim,P[b].max_val, P[b].min_val)
                    P[b].in_plume = False
                    
                N[i].in_PSO = False
                N[i].in_NM = False
                N[i].gbest_fitness = float('-inf')
                N[i].gbest_position = position_initial_random_position(obstacles_info, M.dim, M.map_x_lower, M.map_x_upper, M.map_y_lower, M.map_y_upper)
                # 记录
                N[i].gbest_position_history.append(N[i].gbest_position)
                N[i].gbest_fitness_history.append(N[i].gbest_fitness)   
                
        figure.niche_figure_plot(grid_x, grid_y, uds_t, obstacles_info, P, N, M)
        
    return M.taboo_center, P, N
    
if __name__ == '__main__':
    location, P, N = find_func()  
    for i in range(len(location)):
        print(f"污染源坐标为：{location[i]}")
