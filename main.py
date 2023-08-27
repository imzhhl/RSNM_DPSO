# -*- coding: utf-8 -*-
"""
实现小生境粒子群算法
参考论文：RANGED SUBGROUP PARTICLE SWARM OPTIMIZATION FOR LOCALIZING MULTIPLE ODOR SOURCES

"""
import copy
import numpy as np
import re
from tqdm import tqdm
from fluent2python import Fluent2python
from fitness_wind_function import Fitness, Wind
from obstacle import Obstacle
from figure_plot import Figure_plot
from pn_class import Particle, Niche, Map
import pn_class

# Nelder-Mead算法
'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''
def nelder_mead(f, x_start, step=0.1, no_improve_thr=10e-6, no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)

        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)       #这里是(1,2)，所以dim=1？
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):     
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:                               #这里不应有循环了
        # order
        res.sort(key=lambda x: x[1])       #sort是升序排列，找最小值
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:     #这里max_iter=0，所以不会结束一直循环？
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        # print ('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid 求前面的中心位置
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue                            #continue要改成return了

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres
        
# 改进求斥力
def repulsion_force_2(P, N, M):
    # 计算粒子间的相互距离, p_dist的形状为particle_num*particle_num
    p_dist = np.zeros([pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            p_dist[j, i] = np.linalg.norm(P[i].position - P[j].position)

    # 计算小生境之间的相互距离
    n_dist = np.zeros([pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            n_dist[j, i] = np.linalg.norm(N[i].center_position - N[j].center_position)
    
    # 计算粒子间单位向量, 方向终点减起点
    p_unit_vector = np.zeros([2, pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            if i == j:
                p_unit_vector[:, j, i] = 0
            else:
                p_unit_vector[:, j, i] = ((P[i].position - P[j].position)/np.linalg.norm(P[i].position - P[j].position))[0][:]
     
    # 计算小生境间单位向量, 方向终点减起点
    n_unit_vector = np.zeros([2, pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            if i == j:
                n_unit_vector[:, j, i] = 0
            else:
                n_unit_vector[:, j, i] = ((N[i].center_position - N[j].center_position)/np.linalg.norm(N[i].center_position - N[j].center_position))[0][:]
          
    # 计算粒子间的作用力, 形状为2*size*size, 因为力是向量
    force_particle = np.zeros([2, pn_class.particle_num, pn_class.particle_num])   
    for m in range(pn_class.particle_num):
        for n in range(pn_class.particle_num):
            # 粒子之间的斥力 
            if p_dist[n, m] <= P[m].rcore + P[n].rcore and m != n:
                force_particle[:, n, m] = P[m].coulomb * P[n].coulomb * p_unit_vector[:, n, m] /  P[m].rcore**2
            if P[m].rcore + P[n].rcore < p_dist[n, m] <= P[m].rperc + P[n].rperc and m != n:
                force_particle[:, n, m] = P[m].coulomb * P[n].coulomb * p_unit_vector[:, n, m] / (p_dist[n, m]**2)
            if p_dist[n, m] > P[m].rperc + P[n].rperc and m != n:
                force_particle[:, n, m] = 0
    
    # 计算小生境间的作用力, 形状为2*size*size, 因为力是向量
    force_niche =  np.zeros([2, pn_class.niche_num, pn_class.niche_num])  
    for m in range(pn_class.niche_num):
        for n in range(pn_class.niche_num):
            # 小生境之间的斥力 
            if n_dist[n, m] <= N[m].r_repulse + N[n].r_repulse and m != n:
                force_niche[:, n, m] = n_unit_vector[:, n, m] / n_dist[n, m]
            
    # 求粒子所受合力, 向量相加, 对force沿第二维度自相加到最上面一行
    for i in range(1, force_particle.shape[1]):
        force_particle[:,0,:]=force_particle[:,0,:] + force_particle[:,i,:]
        
    # 粒子所受合力   
    total_force_particle = np.zeros([pn_class.particle_num, M.dimension])
    for i in range(pn_class.particle_num):
        total_force_particle[i, :] = force_particle[:, 0, i]
        
    # 求小生境所受合力, 向量相加, 对force沿第二维度自相加到最上面一行
    for i in range(1, force_niche.shape[1]):
        force_niche[:,0,:]=force_niche[:,0,:] + force_niche[:,i,:]        
    
    # 将小生境受力分配到粒子上去
    for i in range(pn_class.niche_num):
        for menber in N[i].menbers:
            a = re.findall("\d+",menber)
            b = int(a[0])
            total_force_particle[b, :] += force_niche[:, 0, i]
            
    return total_force_particle       
                
#求斥力ai
def repulsion_force(P):
    # 实现niche-pso的功能,主要有以下特点
    # 机制1：从一个小生境到另一小生境的转移机制, 只有非骨干机器人在另一骨干机器人Rcore范围内时才转移分组
    # 机制2：每个小生境都具有一个唯一骨干粒子，具有分发信息、接收数据、管理成员的功能
    # 机制3：不同骨干机器人之间会产生斥力, 骨干机器人和中性/带电机器人之间没有斥力
    # 机制4：带电粒子之间会产生斥力
    # 机制5: 不同小生境之间的pso寻优过程互不干涉

    # 计算粒子间的相互距离, p_dist的形状为particle_num*particle_num
    p_dist = np.zeros([pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            p_dist[j, i] = np.linalg.norm(P[i].position - P[j].position)

    # 计算xi-xp, 结果为向量, 方向终点减起点
    p_unit_vector = np.zeros([2, pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            if i == j:
                p_unit_vector[0, j, i] = 0
                p_unit_vector[1, j, i] = 0
            else:
                p_unit_vector[0, j, i] = ((P[i].position - P[j].position)/np.linalg.norm(P[i].position - P[j].position))[0][0]
                p_unit_vector[1, j, i] = ((P[i].position - P[j].position)/np.linalg.norm(P[i].position - P[j].position))[0][1]
               
            
    # 计算粒子间的作用力, a为向量, 形状为2*size*size, 因为力是向量
    force =  np.zeros([2, pn_class.particle_num, pn_class.particle_num])   
    
    for m in range(pn_class.particle_num):
        for n in range(pn_class.particle_num):
            
            # # 非骨干带电粒子之间的斥力 
            if P[n].main == False and P[m].main == False and p_dist[n, m] <= P[m].rcore and m != n:
                force[:, n, m] = P[m].coulomb * P[n].coulomb * p_unit_vector[:, n, m] /  P[m].rcore**2
            if P[n].main == False and P[m].main == False and P[m].rcore < p_dist[n, m] <= P[m].rperc and m != n:
                force[:, n, m] = P[m].coulomb * P[n].coulomb * p_unit_vector[:, n, m] / (p_dist[n, m]**2)
            if P[n].main == False and P[m].main == False and p_dist[n, m] > P[m].rperc and m != n:
                force[:, n, m] = 0
   
            # 骨干粒子之间的斥力
            if P[n].main == True and P[m].main == True and p_dist[n, m] <= P[m].r_repulse + P[n].r_repulse and m != n:
                force[:, n, m] = 1/(P[m].r_repulse**2 )* p_unit_vector[:, n, m]
            if P[n].main == True and P[m].main == True and p_dist[n, m] >  P[m].r_repulse + P[n].r_repulse and m != n:
                force[:, n, m] = 0
                
            #非带电粒子之间的斥力; 带电粒子与骨干粒子直接的斥力都为0
            
    # 适应度最大的骨干粒子不受斥力, 其他骨干粒子受斥力
    for m in range(pn_class.particle_num):
        for n in range(pn_class.particle_num):
            # m收到斥力, n不受斥力
            if P[n].fitness >  P[m].fitness:
                force[:, m, n] = 0
                
            # n收到斥力, m不受斥力
            if P[n].fitness <= P[m].fitness:
                force[:, n, m] = 0
    
    # 求所受合力, 向量相加, 对force沿第二维度自相加到最上面一行
    for i in range(1, force.shape[1]):
        force[:,0,:]=force[:,0,:] + force[:,i,:]
        
    total_force = np.zeros([pn_class.particle_num, pn_class.dimension])
    for i in range(pn_class.particle_num):
        total_force[i, 0] = force[0, 0, i]
        total_force[i ,1] = force[1, 0, i]

    return total_force

# niche 归属小生境更新(根据距离计算)
def niche_belonged_update(P):
    
    # 计算粒子间的相互距离, p_dist的形状为size*size
    p_dist = np.zeros([pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            p_dist[j, i] = np.linalg.norm(P[i].position - P[j].position)
            
    # 计算粒子与main粒子之间的距离, 如果在捕获半径内, 则更新niche信息
    for i in range(pn_class.particle_num):
        if P[i].main == True:
            niche_id = P[i].belonged_niche_NO
            for j in range(pn_class.particle_num):
                if p_dist[j, i] < P[i].r_attract and P[j].main == False:
                    P[j].belonged_niche_NO = niche_id
    
    return P
                    
##########
# 更新速度，根据公式V(t+1) = w * V(t) + c1 * r1 * (pbest_i - xi) + c2 * r2 * (gbest_xi - xi)
def velocity_update(V, X, wind, force, pbest, gbest, c1, c2, w):
    r1 = np.random.random(1) # 0-1间的随机数
    r2 = np.random.random(1) # 0-1间的随机数

    # 更新速度
    V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X) + force # 一行两列
    
    #乘控制参数, 目的为利用风速风向信息
    # V = V * cp(X, wind, V)
    return V

# wind utilization controled by parameter "cp" (chi_theta)
def cp(X, wind, V):  #控制参数
    # 当风向与机器人运动方向相同时，控制参数cp取得最小值0
    # 当风向与机器人运动方向相反时，控制参数cp取得最大值1
    # 中间过程从0到1连续变化
    
    #算W和V的内积，粒子飞行速度和风场风向的内积
    a = np.dot(wind, V.reshape(2,1))

    # 算wind的模
    wind_norm = np.linalg.norm(wind)
    
    # 算V的模
    V_norm = np.linalg.norm(V)

    # 算W和V之间的角度的余弦值
    # 返回参数前对V_norm进行判断是否为0, 防止返回Nan
    # 此时速度太小了, 可能是因为pso固有的缺点, 在极值点处震荡
    if V_norm == 0:
        cos = 1
    else:
        cos = a/(wind_norm * V_norm)

    chi_theta = 0.5*(1-cos)
    return chi_theta
            
##########
#更新粒子位置，根据公式X(t+1)=X(t)+V, 并约束粒子范围
def position_updata(X, V, radius, center):
    if np.linalg.norm(center - (X + V)) > radius:
        return X                              #约束粒子范围，但这直接打回原形了？
    else:
        return X + V

def taboo_zone_avoid(X, V, M):
    if len(M.taboo_center) != 0: # 如果已经形成了禁忌区
        for i in range(len(M.taboo_center)): # 循环所有禁忌区
            distance = np.linalg.norm(X - M.taboo_center[i])
            # unit_vector = (X - M.taboo_center[i])/np.linalg.norm(X - M.taboo_center[i])
            unit_vector = np.array((np.random.uniform(0,1),np.random.uniform(0,1))).reshape(1,2)
            
            while distance < M.taboo_radius:
                X = X + 100*unit_vector                               #当遇到搜索禁区的时候，就再随机一个新的方向吗？
                distance = np.linalg.norm(X - M.taboo_center[i])     

    return X

def position_initial(obstacles_info, dim, map_x_lower, map_x_upper, map_y_lower, map_y_upper):
    X = np.zeros([1, dim])
    X[0,0] = np.random.uniform(map_x_lower,map_x_upper,size= 1) # x坐标(0, 2000)
    X[0,1] = np.random.uniform(map_y_lower,map_y_upper,size= 1) # y坐标(0, 1000)
    ix = np.rint(X[0,0]).astype(int)
    iy = np.rint(X[0,1]).astype(int)
    while obstacles_info[iy,ix] == 1:
        X[0,0] = np.random.uniform(map_x_lower,map_x_upper,size=1) # x坐标(0, 2000)
        X[0,1] = np.random.uniform(map_y_lower,map_y_upper,size=1) # y坐标(0, 1000)
        ix = np.rint(X[0,0]).astype(int)
        iy = np.rint(X[0,1]).astype(int)
    return X

# 计算小生境的中心
def niche_center(N, P):
    for i in range(pn_class.niche_num):
        neutral_p = []
        for menber in N[i].menbers:
            a = re.findall("\d+",menber)
            b = int(a[0])
            if P[b].coulomb == 0:                      #这样计算的是中性粒子的位置中心
                neutral_p.append(P[b].position[0])
        neutral_p = np.array(neutral_p)

        xx = np.average(neutral_p[:,0])
        yy = np.average(neutral_p[:,1])
        N[i].center_position = np.array([[xx, yy]])   #(1,2)
    return N
    
# 初始化位置
def position_initial_2(i): 
    if i == 0:
        return np.array([[200,200]])
    if i == 1:
        return np.array([[200,100]])
    if i == 2:
        return np.array([[300,200]])
    if i == 3:
        return np.array([[300,100]])
    if i == 4:
        return np.array([[400,200]])
    if i == 5:
        return np.array([[400,100]])
    if i == 6:
        return np.array([[500,200]])
    if i == 7:
        return np.array([[500,100]])

def boundary_handle(X, M):
    if X[0,0] < M.map_x_lower:
        X[0,0] = M.map_x_lower
    
    if X[0,0] > M.map_x_upper:
        X[0,0] = M.map_x_upper
    
    if X[0,1] < M.map_y_lower:
        X[0,1] = M.map_y_lower
        
    if X[0,1] > M.map_y_upper:
        X[0,1] = M.map_y_upper
        
    return X

def velocity_boundary_handle(V, max_val, min_val):       
    V[V < min_val] = min_val
    V[V > max_val] = max_val
    return V

def velocity_initial(dim, max_val, min_val):
    velocity = np.random.uniform(min_val,max_val,size=(1, dim))
    return velocity

def f_nm(x):
    return -func.fitness_func(x.reshape(1,2), uds_t)
    
########

#%%  一些预处理参数
M = Map() # 地图实例化
func = Fitness()                # 实例化适应度类
wind = Wind()                   # 实例化风向类
obstacle = Obstacle()           # 实例化障碍物类
fluent2python = Fluent2python() # 实例化fluent数据类
uds_t, wind_t, grid_x, grid_y = fluent2python.file_to_array()     # 运行一次就够了
obstacles_info = obstacle.obstacle_define(M.obstacle_list) # 获取障碍物数组 
figure = Figure_plot()          # 绘图类实例化

# figure.figure_plot_1(P, obstacles_info)

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
    P[i].position = position_initial_2(i)
    # P[i].position = position_initial(obstacles_info, M.dimension, M.map_x_lower, M.map_x_upper, M.map_y_lower, M.map_y_upper)
    P[i].velocity = velocity_initial(M.dimension, P[i].max_val, P[i].min_val)
    # 设置骨干机器人, 第0和2为骨干机器人
    # if i == 0 or i == 1:
    #     P[i].main = True
    # else:
    #     P[i].main = False
    # 设置小生境 0, 1, 2, 3 属第0个小生境
    #          4, 5, 6, 7 属第1个小生境
    if i == 0 or i == 1 or i == 2 or i == 3 :
        P[i].belonged_niche_NO = 0
    else:
        P[i].belonged_niche_NO = 1
        
    # 设置电荷属性, 骨干机器人不需要考虑电荷属性, 第1个和第5个为neutral, 其他为charged
    if i == 0 or i == 1 or i == 4 or i == 5:
        P[i].coulomb = 0
    else:
        P[i].coulomb = 1
    
    #所有粒子都是neutral
    # P[i].coulomb = 0

# 初始化
# 更新粒子信息
for i in range(pn_class.particle_num):
    P[i].fitness = func.fitness_func(P[i].position, uds_t)
    P[i].pbest_fitness = P[i].fitness    # 初始化
    P[i].pbest_position = P[i].position  # 初始化
  
for i in range(pn_class.niche_num):
    N[i].gbest_position = P[i].position  #种群的全局历史最优值有问题,不可能是N[0]=P[0],N[1]=P[1]                        
    
# %%
################################################################################## 迭代 ##################################################################################
for iter_num in tqdm(range(pn_class.total_iter)):
    # 更新小生境信息
    for i in range(pn_class.niche_num):
        N[i].menbers = []
        N[i].UAVs_position = dict()
        N[i].UAVs_fitness = dict()
        for j in range(pn_class.particle_num):
            if P[j].belonged_niche_NO == N[i].NO:
                N[i].menbers.append('UAVs_'+str(j))
                N[i].UAVs_position['UAVs_'+str(j)] = P[j].position
                N[i].UAVs_fitness['UAVs_'+str(j)] = P[j].fitness
    
    # 并更新粒子所属小生境
    # P = niche_belonged_update(P)
    
    # 计算小生境的中心
    N = niche_center(N, P)
    
    # 每个粒子所受的力
    total_force = repulsion_force_2(P, N, M) # 斥力
            
    # 更新粒子信息
    for i in range(pn_class.particle_num):
        P[i].force = total_force[i].reshape(1,2)
        P[i].wind = wind.wind_func(P[i].position, wind_t)
        P[i].fitness_history.append(P[i].fitness)
        P[i].position_history.append(P[i].position)
        

    # 更新粒子的速度, one by one for particles
    for i in range(pn_class.niche_num):
        if N[i].actived == True:
            max_pbest_fitness2 = float('-inf')
            max_pbest_index = float('-inf')
            for menber in N[i].menbers:
                a = re.findall("\d+",menber)
                b = int(a[0])
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
                P[b].position = taboo_zone_avoid(P[b].position, P[b].velocity, M)     
                P[b].position = boundary_handle(P[b].position, M)
                P[b].fitness = func.fitness_func(P[b].position, uds_t)         #这里感觉有问题，不应该对上一步的fitness改动
                P[b].fitness_history.append(P[b].fitness)
                P[b].pbest_fitness2 = func.fitness_func(P[b].position, uds_t)
                
                
                # 寻找所有粒子更新后的最大适应度, 赋值给所属小生境gbest_fitness2
                if P[b].pbest_fitness2 >= max_pbest_fitness2:
                    max_pbest_fitness2 = P[b].pbest_fitness2
                    max_pbest_position2 = P[b].position
                    max_pbest_index = b
                N[i].gbest_fitness2 = max_pbest_fitness2
            
        # 更新骨干粒子信息    
        # for menber in N[i].menbers:
        #     a = re.findall("\d+",menber)
        #     b = int(a[0])
        #     P[b].main = False 
        # P[max_pbest_index].main = True 
                
        # 更新每个粒子的历史最优位置和种群的全局最优位置
        for menber in N[i].menbers:
            a = re.findall("\d+",menber)
            b = int(a[0])
            if P[b].pbest_fitness < P[b].pbest_fitness2:
                P[b].pbest_position = P[b].position
                P[b].pbest_fitness = P[b].pbest_fitness2
    
            if N[i].gbest_fitness < N[i].gbest_fitness2:
                N[i].gbest_position = max_pbest_position2
                N[i].gbest_fitness = N[i].gbest_fitness2
    
    # 更新粒子的速度, one by one for particles
    for i in range(pn_class.niche_num):        #关闭pso，打开nm算法
        if N[i].actived == False:
            source_position = nelder_mead(f_nm, N[i].center_position)[0][0]
            
    
    for i in range(pn_class.niche_num):           #有些重复
        # 小生境中适应度值最大的无人机的索引
        i_gbest = max(N[i].UAVs_fitness, key=lambda x:N[i].UAVs_fitness[x])
        N[i].gbest_fitness = N[i].UAVs_fitness[i_gbest]
        N[i].gbest_position = N[i].UAVs_position[i_gbest]
        N[i].gbest_position_history.append(N[i].gbest_position.tolist()[0])
        N[i].gbest_fitness_history.append(N[i].gbest_fitness) 
    
    # 计算小生境的聚集度     
    for i in range(pn_class.niche_num):      #这里是否需要分段计算聚集度
        s_sum = 0 # 初始化相似度
        for menber in N[i].menbers:
            s = 1 - np.linalg.norm(N[i].UAVs_position[menber] - N[i].gbest_position) / N[i].niche_Gr
            s_sum = s_sum + s
        N[i].agregation = s_sum / len(N[i].menbers)
           
    # 如果聚集度大于0.3则开始进行Nelder-Mead算法
    for i in range(pn_class.niche_num):
        if N[i].actived == True:
            # 三个判断条件
            print(N[i].agregation)
            print(N[i].gbest_fitness)
            if  N[i].agregation > N[i].epsilon_nm and N[i].gbest_fitness > 0.07: #N[i].C_threshold_nm:   #这里如果更换数据，相应阈值也要变化
                N[i].actived = False
    
    # 设置搜索禁忌区
    # for i in range(pn_class.niche_num):
    #     if N[i].actived == True:
    #         # 三个判断条件
    #         if  N[i].agregation > N[i].epsilon and N[i].gbest_fitness > N[i].C_threshold:
    #             M.taboo_center.append(N[i].gbest_position)
    # print(N[i].gbest_position_history)    
    figure.niche_figure_plot(grid_x, grid_y, uds_t, wind_t, obstacles_info, P, N)
            
for i in range(pn_class.niche_num):
    print(f"N{i} 最优适应度：{N[i].gbest_fitness}")
    print(f"N{i} 最优位置：x={N[i].gbest_position[0][0]},y={N[i].gbest_position[0][1]}")
    print("-----------------------")

# 绘图
# figure.figure_plot(grid_x, grid_y, uds_t, wind_t, obstacles_info, P, N)