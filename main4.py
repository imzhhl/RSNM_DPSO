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
                            merge_arr_within_distance, boundary_handle_nm, position_initial_fixed_position_around

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
    boundary_handle_nm(M, xr)                       # 防超域处理
    rscore = func.fitness_func(xr, uds_t)  
    
      
    if P[m[-2]].fitness <= rscore < P[m[0]].fitness:# 如果对称点的值不大不小
        # P[-1].position = NM_obstacle_avoid(P[-1].position, xr)   # 直接将对称点替换掉最小的点
        P[m[-1]].position = xr
        P[m[-1]].fitness = func.fitness_func(P[m[-1]].position, uds_t)
        return P,N                               # 后面不运行了，进入下一次循环

    # expansion
    elif rscore > P[m[0]].fitness:               # 如果对称点的值最大（说明这个方向是对的，所以多加点）
        xe = x0 + gamma*(x0 - P[m[-1]].position) # gamma=2.   求最小值点关于其他点平均值点更远的对称点
        boundary_handle_nm(M, xe)                   # 防超域处理
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
        boundary_handle_nm(M, xc)                   # 防超域处理
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
                boundary_handle_nm(M, tup.position)                      # 防超域处理
                tup.fitness = func.fitness_func(tup.position, uds_t)  # 将之前所有点都关于最小值点对称过去，重新组成新的点
    
    N.gbest_fitness = P[m[0]].fitness
    N.gbest_position = P[m[0]].position

                            
def find_func(total_iter, robots_number, niche_radius, response_time, errorness):
    # 真实污染源位置
    source_1 = np.array([70.7, 193.8])
    source_2 = np.array([124.5, 263.7])
    distance_error = 50
    source_1_check = []
    source_2_check = []
    
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
    
    grid_x = np.load("F:/ZHHL/TE_Doctor/研究内容/SCI论文/8-粒子群算法主动溯源/Niche-DPSO/data/input_data/1200-5-sin/grid_x.npy")
    grid_y = np.load("F:/ZHHL/TE_Doctor/研究内容/SCI论文/8-粒子群算法主动溯源/Niche-DPSO/data/input_data/1200-5-sin/grid_y.npy")
    uds_t  = np.load('F:/ZHHL/TE_Doctor/研究内容/SCI论文/8-粒子群算法主动溯源/Niche-DPSO/data/input_data/1200-5-sin/uds_t_1.npy')
    
    # 小生境实例化
    N = []
    for i in range(pn_class.niche_num):
        N.append(Niche())
        N[i].NO = i
        
    # 粒子实例化
    P = []
    for i in range(robots_number):
        P.append(Particle())
        P[i].NO = i
        # 小生境划分,前一半属于小生境0,后一半属于小生境1
        if 0<= i < robots_number/2 :
            P[i].belonged_niche_NO = 0
        else:
            P[i].belonged_niche_NO = 1
            
        # P[i].position = position_initial_fixed_position_around(i) # 固定位置初始化
        P[i].position = position_initial_random_position(M.dim, M.map_x_lower, M.map_x_upper, M.map_y_lower, M.map_y_upper) # 随机位置初始化
        P[i].velocity = velocity_initial(M.dim, P[i].max_val, P[i].min_val) # 速度初始化
        P[i].unit_vector = normalize(P[i].velocity)
        P[i].fitness = func.fitness_func(P[i].position, uds_t)
    
        # 记录
        P[i].fitness_history.append(P[i].fitness)
        P[i].position_history.append(P[i].position)    
    
    # 初始化
    # 更新粒子信息
    for i in range(robots_number):
        # P[i].fitness = func.fitness_func(P[i].position, uds_t)
        P[i].pbest_fitness = P[i].fitness    # 初始化
        P[i].pbest_position = P[i].position  # 初始化
      
    # 初始化全局最优值
    for i in range(pn_class.niche_num):
        N[i].menbers = []
        for j in range(robots_number):
            if P[j].belonged_niche_NO == N[i].NO:
                N[i].menbers.append('UAVs_'+str(j))
        for menber in N[i].menbers:        
            b = int(re.findall("\d+",menber)[0])
        
        N[i].gbest_fitness =  float('-inf')
        N[i].gbest_position = np.array([[float('-inf'),float('-inf')]])
        # 记录
        N[i].gbest_position_history.append(N[i].gbest_position)
        N[i].gbest_fitness_history.append(N[i].gbest_fitness)
    
    #% 迭代过程
    exit_flag  = False # 用于直接跳出主循环用的
    success = False  # 用于判断是否溯源成功
    # %% 开始循环
    for iter_num in range(1, total_iter+1):
        uds_t_0  = np.load(f'F:/ZHHL/TE_Doctor/研究内容/SCI论文/8-粒子群算法主动溯源/Niche-DPSO/data/input_data/1200-5-sin/uds_t_{iter_num*response_time}.npy')
        uds_t =  uds_t_0*(1+np.random.uniform(-errorness, errorness))
        wind_t = np.load(f'F:/ZHHL/TE_Doctor/研究内容/SCI论文/8-粒子群算法主动溯源/Niche-DPSO/data/input_data/1200-5-sin/wind_t_{iter_num*response_time}.npy')
        
        if exit_flag:
            break
        
        #---------------检测列表是否反生变化
        list_new = M.taboo_center # 判断列表是否发生变化
        if list_new != list_old:
            taboo_zone = taboo_zone_cal(M)
            list_old = list_new.copy()
        #--------------- 
        # =========================================================================
        # 第一阶段, 羽流发现         
        # ========================================================================= 
        
        for i in range(robots_number):
            # 计算每个无人机适应度
            P[i].fitness = func.fitness_func(P[i].position, uds_t)
            # 躲避禁区
            P[i].fitness = taboo_zone_avoid(P[i].position, P[i].fitness, taboo_zone)
            # 将新适应度增加到列表中
            P[i].position_history.append(P[i].position)
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
                P[i].position, P[i].velocity = boundary_handle(M, P[i].position, P[i].velocity)
                P[i].position_history.append(P[i].position)
                # 躲避禁区
                P[i].fitness = taboo_zone_avoid(P[i].position, P[i].fitness, taboo_zone)
                # 碰壁检测
                P[i].rebound = check_boundary(M, P[i].position)
    
                # 如果前进后碰壁, 则回退一步, 并进行20米的反弹
                if P[i].rebound == True:
                    P[i].unit_vector = -P[i].unit_vector
                    P[i].position = step_func(P[i].position, P[i].unit_vector, P[i].step)
                    P[i].position_history.append(P[i].position)
                    # 再随机一个初始方向
                    # V = random.uniform(size=(1,robots.UAVs.dimension))
                    # U[i].unit_vector = normalize(V)
                    
                    # 再旋转45度弹出
                    P[i].unit_vector = turn_45_vector(P[i].unit_vector) 
                    
        # 循环所有的无人机
        for i in range(robots_number):     # 只要有一个检测到污染羽流，整个种群就开始PSO
            if P[i].in_plume == True:
                j = P[i].belonged_niche_NO                    
                N[j].in_PSO = True
    
        # 循环所有的小生境, 更新小生境中的参数
        for i in range(pn_class.niche_num):
            # 更新所有小生境中各个字典的信息
            for menber in N[i].menbers:
                b = int(re.findall("\d+",menber)[0])
                N[i].UAVs_position['UAV_' + str(b)] = P[b].position       # UAVs_position应该是每步变化的
                N[i].UAVs_fitness['UAV_' + str(b)] = P[b].fitness         # UAVs_fitness应该是每步变化的
                N[i].UAVs_unit_vector['UAV_' + str(b)] = P[b].unit_vector # UAVs_unit_vector应该是每步变化的 
    
            # 小生境中适应度值最大的无人机的索引
            i_gbest = max(N[i].UAVs_fitness, key=lambda x:N[i].UAVs_fitness[x])
            N[i].gbest_fitness = N[i].UAVs_fitness[i_gbest]
            N[i].gbest_position = N[i].UAVs_position[i_gbest]
            N[i].gbest_unit_vector = N[i].UAVs_unit_vector[i_gbest]
    
        # =========================================================================
        # 第二阶段, 羽流追踪        
        # =========================================================================             
        # 接入PSO, 每个niche独立进行PSO         
        
        # 计算小生境的中心
        N = niche_center(N, P)
        for i in range(pn_class.niche_num):
            N[i].center_position_history.append(N[i].center_position)
    
        # 每个粒子所受的力
        total_force = repulsion_force_path(P, N, M, niche_radius, robots_number) # 利用路径方法求斥力
        for i in range(robots_number):
            P[i].force = total_force[i].reshape(1,2)
        
        # 更新粒子的速度, one by one for particles
        for i in range(pn_class.niche_num):
            # 计算小生境的聚集度 
            dmin = 10
            dmax = 100    
            s_sum = 0 # 初始化相似度
            for menber in N[i].menbers:
                b = int(re.findall("\d+",menber)[0])
                d = np.linalg.norm(P[b].position - N[i].gbest_position)
                if d <= dmin:
                    s = 1
                elif dmin < d < dmax:
                    s = 1 - d / niche_radius
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
                for menber in N[i].menbers:
                    b = int(re.findall("\d+",menber)[0])
                    P[b].wind = wind.wind_func(P[b].position, wind_t)
                    # 更新小生境每个无人机的速度
                    P[b].velocity = velocity_update(P[b].velocity,
                                                    P[b].position, 
                                                    P[b].wind, 
                                                    P[b].force, 
                                                    P[b].pbest_position, 
                                                    N[i].gbest_position,
                                                    P[b].c_ini_1,
                                                    P[b].c_end_1,
                                                    P[b].c_ini_2,
                                                    P[b].c_end_2,
                                                    P[b].w,
                                                    iter_num,
                                                    total_iter)
                    # 限制速度大小
                    P[b].velocity = velocity_boundary_handle(P[b].velocity, P[b].max_val, P[b].min_val)
                    # 更新粒子位置
                    P[b].position = position_updata(P[b].position, P[b].velocity)
                    # 限制粒子位置
                    P[b].position, P[b].velocity = boundary_handle(M, P[b].position, P[i].velocity)
                    # 将新位置增加到列表中
                    P[b].position_history.append(P[b].position)
                    # 计算更新后粒子的适应度
                    P[b].fitness = func.fitness_func(P[b].position, uds_t)
                    # 躲避禁区
                    P[b].fitness = taboo_zone_avoid(P[b].position, P[b].fitness, taboo_zone)
                    # 将新适应度增加到列表中
                    P[b].fitness_history.append(P[b].fitness)
                    
                    # 更新pbest_fitness
                    if P[b].fitness >= P[b].pbest_fitness:
                        P[b].pbest_fitness = P[b].fitness
                        P[b].pbest_position = P[b].position
                        
                    
                    # 更新gbest_fitness
                    if P[b].fitness >= N[i].gbest_fitness:
                        N[i].gbest_fitness = P[b].fitness
                        N[i].gbest_position = P[b].position
                # 记录            
                N[i].gbest_position_history.append(N[i].gbest_position)
                N[i].gbest_fitness_history.append(N[i].gbest_fitness)
            
            # 开始NM算法
            if N[i].in_PSO == True and N[i].in_NM == True:
                # print(f"NM{i}")
                nelder_mead(P, N[i], M, uds_t, func)
                # 记录
                N[i].gbest_position_history.append(N[i].gbest_position)
                N[i].gbest_fitness_history.append(N[i].gbest_fitness)
            
        for i in range(robots_number):
            # 记录
            P[i].fitness_history.append(P[i].fitness)
            P[i].position_history.append(P[i].position)
            
        # =========================================================================
        # 第三阶段, 羽流确认        
        # =========================================================================                              
        # 某个小生境的全局极值在一定迭代步数内变化不大,聚集度够大,适应度够大,则说明已经定位成功 N[i].agregation > N[i].epsilon and
        # 形成搜索禁区         
            
        for i in range(pn_class.niche_num):
            # 三个判断条件 N[i].agregation > N[i].epsilon_end and
            if  N[i].gbest_fitness > N[i].C_threshold_end:
                M.taboo_center.append([N[i].gbest_position[0][0],N[i].gbest_position[0][1]])
                # 合并距离较近的两个源
                M.taboo_center = merge_arr_within_distance(M.taboo_center, 20)
                
                for menber in N[i].menbers:
                    b = int(re.findall("\d+",menber)[0])
                    P[b].fitness = float('-inf')
                    P[b].pbest_fitness = P[b].fitness
                    # P[b].position = position_initial_fixed_position_around(b)
                    P[b].position = position_initial_random_position( M.dim, M.map_x_lower, M.map_x_upper, M.map_y_lower, M.map_y_upper)
                    P[b].pbest_position = P[b].position
                    P[b].in_plume = False
                    
                N[i].in_PSO = False
                N[i].in_NM = False
                N[i].gbest_fitness = float('-inf')
                
        # if N[0].in_PSO == False and N[1].in_PSO == False:
        #     break
    
        location = np.array(M.taboo_center)
        
        for j in range(len(location)):
            source_1_check.append(np.linalg.norm(source_1 - location[j]))
            source_2_check.append(np.linalg.norm(source_2 - location[j]))
        
        if len(source_1_check)>0 and len(source_2_check)>0:
            if min(source_1_check) < distance_error and min(source_2_check) < distance_error:
                success = True
                exit_flag = True
                
    # figure.niche_figure_plot(grid_x, grid_y, uds_t, obstacles_info, P, N, M)
    return iter_num, success
    
if __name__ == '__main__':
    iter_num, success, plume_finding = find_func()
    print(f"跑了多少步？{iter_num}，成功了吗？：{success},羽流发现需要{plume_finding}步")
