import numpy as np
import re
from fluent2python import Fluent2python
from fitness_wind_function import Fitness, Wind
from obstacle import Obstacle
from figure_plot import Figure_plot
from pn_class import Particle, Niche, Map
import pn_class

def P_menber(menbers):
    m = []
    for i in range(len(menbers)):
        m.append(int(re.findall("\d+", menbers[i])[0]))
    return m

def nelder_mead(P, N, M, no_improve_thr=10e-6, no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    
    m = P_menber(N.menbers)
    
    # init
    # 更新所有粒子的适应值
    for i in m:
        P[i].fitness = func.fitness_func(P[i].position, uds_t)
        # print(P[i].fitness)
        
    # 降序排列,只需要将N.menber里面的内容排序即可
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if P[m[i]].fitness < P[m[j]].fitness:
                N.menbers[i], N.menbers[j] = N.menbers[j], N.menbers[i]
    print(N.menbers)
    # print(P[m[0]].position, P[m[0]].fitness)
    # print(P[m[1]].position, P[m[1]].fitness)
    # print(P[m[2]].position, P[m[2]].fitness)
    # print(P[m[3]].position, P[m[3]].fitness)

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
    boundary_handle(xr, M)                       # 防超域处理
    rscore = func.fitness_func(xr, uds_t)  
    
      
    if P[m[-2]].fitness <= rscore < P[m[0]].fitness:# 如果对称点的值不大不小
        # P[-1].position = NM_obstacle_avoid(P[-1].position, xr)   # 直接将对称点替换掉最小的点
        P[m[-1]].position = xr
        P[m[-1]].fitness = func.fitness_func(P[m[-1]].position, uds_t)
        return P,N                               # 后面不运行了，进入下一次循环

    # expansion
    elif rscore > P[m[0]].fitness:               # 如果对称点的值最大（说明这个方向是对的，所以多加点）
        xe = x0 + gamma*(x0 - P[m[-1]].position) # gamma=2.   求最小值点关于其他点平均值点更远的对称点
        boundary_handle(xe, M)                   # 防超域处理
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
        boundary_handle(xc, M)                   # 防超域处理
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
                boundary_handle(tup.position, M)                      # 防超域处理
                tup.fitness = func.fitness_func(tup.position, uds_t)  # 将之前所有点都关于最小值点对称过去，重新组成新的点
    
    N.gbest_fitness = P[m[0]].fitness
    N.gbest_position = P[m[0]].fitness

def taboo_zone_avoid_force(X, V, M, taboo_zone):
    if taboo_zone[int(X[0][1])-2,int(X[0][0])-2] == 1:
        V = -V
        X = X + 10*V 
    return X, V

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

# 利用路径求斥力
def repulsion_force_path(P, N, M):
    # 计算粒子间的相互距离, p_dist的形状为particle_num*particle_num
    p_dist = np.zeros([pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            p_dist[i, j] = np.linalg.norm(P[i].position - P[j].position)

    # 计算小生境之间的相互距离, n_dist的形状为niche_num*niche_num
    niche_dist = np.zeros([pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            niche_dist[i, j] = np.linalg.norm(N[i].gbest_position - N[j].gbest_position)
            
    # 计算当前小生境位置距另一小生境历史位置的最近距离
    path_dist = np.zeros([pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            min_dist = np.linalg.norm((N[i].gbest_position - N[j].gbest_position_history), axis=2)
            path_dist[i, j] = np.linalg.norm(N[i].gbest_position-N[j].gbest_position_history[np.argmin(min_dist)])
              
    # 计算粒子间单位向量, 方向终点减起点
    p_unit_vector = np.zeros([M.dim, pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            if i == j:
                p_unit_vector[:, i, j] = 0
            else:
                p_unit_vector[:, i, j] = ((P[j].position - P[i].position)/np.linalg.norm(P[j].position - P[i].position))
     
    # 计算小生境间单位向量, 方向终点减起点
    niche_unit_vector = np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            if i == j:
                niche_unit_vector[:, i, j] = 0
            else:
                niche_unit_vector[:, i, j] = ((N[j].gbest_position - N[i].gbest_position)/np.linalg.norm(N[j].gbest_position - N[i].gbest_position))

    # 计算小生境与路径间的单位向量, 方向终点减起点
    path_unit_vector = np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            min_dist = np.linalg.norm((N[i].gbest_position - N[j].gbest_position_history), axis=2)
            if i == j:
                path_unit_vector[:, i, j] = 0
            else:
                path_unit_vector[:, i, j] = ((N[j].gbest_position_history[np.argmin(min_dist)] - N[i].gbest_position)/np.linalg.norm(N[j].gbest_position_history[np.argmin(min_dist)] - N[i].gbest_position))
     
    # 计算粒子间的作用力, 形状为2*size*size, 因为力是向量
    force_particle = np.zeros([M.dim, pn_class.particle_num, pn_class.particle_num])   
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            # 粒子之间的斥力 
            if p_dist[i, j] <= P[i].rcore + P[j].rcore and i != j:
                force_particle[:, i, j] = P[i].coulomb * P[j].coulomb * p_unit_vector[:, i, j] /  P[i].rcore**2
            if P[i].rcore + P[j].rcore < p_dist[i, j] <= P[i].rperc + P[j].rperc and i != j:
                force_particle[:, i, j] = P[i].coulomb * P[j].coulomb * p_unit_vector[:, i, j] / p_dist[i, j]**2
            if p_dist[i, j] > P[i].rperc + P[j].rperc and i != j:
                force_particle[:, i, j] = 0
    
    # 1.求粒子所受合力
    force_particle_summed = np.sum(force_particle, axis=2, keepdims=True)
    
    # 计算小生境间的即时斥力, 形状为2*size*size, 因为力是向量
    force_niche_immediate = np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])  
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            # 小生境之间的即时斥力 
            if niche_dist[i, j] <= N[i].r_repulse + N[j].r_repulse and i != j:
                force_niche_immediate[:, i, j] = niche_unit_vector[:, i, j] / niche_dist[i, j]**2
    
    # 计算小生境间的路径斥力, 形状为2*size*size, 因为力是向量
    force_niche_path =  np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])  
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            # 小生境之间的路径斥力 
            if path_dist[i, j] <= N[i].r_repulse + N[j].r_repulse and i != j:
                force_niche_path[:, i, j] = path_unit_vector[:, i, j] / path_dist[i, j]**2       
   
    # 2.求小生境所受即时即时斥力合力
    force_niche_immediate_summed = np.sum(force_niche_immediate, axis=2, keepdims=True)

    # 3.求小生境所受路径斥力合力
    force_niche_path_summed = np.sum(force_niche_path, axis=2, keepdims=True)         
            
    # 粒子所受合力, 将小生境受力分配到粒子上去
    total_force_particle = np.zeros([pn_class.particle_num,M.dim])
    for i in range(pn_class.niche_num):
        for menber in N[i].menbers:
            a = re.findall("\d+",menber)
            b = int(a[0])
            total_force_particle[b, :] = (force_particle_summed[:, b, :] + force_niche_immediate_summed[:, i, :] + 100*force_niche_path_summed[:, i, :]).T
            
    return total_force_particle       

    
# 改进求斥力
def repulsion_force_2(P, N, M):
    
    # 计算粒子间的相互距离, p_dist的形状为particle_num*particle_num
    p_dist = np.zeros([pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            p_dist[j, i] = np.linalg.norm(P[i].position - P[j].position)

    # 计算小生境之间的相互距离, n_dist的形状为niche_num*niche_num
    n_dist = np.zeros([pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            n_dist[j, i] = np.linalg.norm(N[i].center_position - N[j].center_position)
    
    # 计算粒子间单位向量, 方向终点减起点
    p_unit_vector = np.zeros([M.dim, pn_class.particle_num, pn_class.particle_num])
    for i in range(pn_class.particle_num):
        for j in range(pn_class.particle_num):
            if i == j:
                p_unit_vector[:, j, i] = 0
            else:
                p_unit_vector[:, j, i] = ((P[i].position - P[j].position)/np.linalg.norm(P[i].position - P[j].position))
     
    # 计算小生境间单位向量, 方向终点减起点
    n_unit_vector = np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])
    for i in range(pn_class.niche_num):
        for j in range(pn_class.niche_num):
            if i == j:
                n_unit_vector[:, j, i] = 0
            else:
                n_unit_vector[:, j, i] = ((N[i].center_position - N[j].center_position)/np.linalg.norm(N[i].center_position - N[j].center_position))

    # 计算粒子间的作用力, 形状为2*size*size, 因为力是向量
    force_particle = np.zeros([M.dim, pn_class.particle_num, pn_class.particle_num])   
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
    force_niche =  np.zeros([M.dim, pn_class.niche_num, pn_class.niche_num])  
    for m in range(pn_class.niche_num):
        for n in range(pn_class.niche_num):
            # 小生境之间的斥力 
            if n_dist[n, m] <= N[m].r_repulse + N[n].r_repulse and m != n:
                force_niche[:, n, m] = n_unit_vector[:, n, m] / n_dist[n, m]        #公式出处？
            
    # 求粒子所受合力, 向量相加, 对force沿第二维度自相加到最上面一行
    for i in range(1, force_particle.shape[1]):
        force_particle[:,0,:]=force_particle[:,0,:] + force_particle[:,i,:]
        
    # 粒子所受合力   
    total_force_particle = np.zeros([pn_class.particle_num, M.dim])
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

def position_initial_random_position(obstacles_info, dim, map_x_lower, map_x_upper, map_y_lower, map_y_upper):
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
# 通过对所有粒子的位置取均值获得
def niche_center(N, P):
    for i in range(pn_class.niche_num):
        neutral_p = []
        for menber in N[i].menbers:
            a = re.findall("\d+",menber)
            b = int(a[0])
            neutral_p.append(P[b].position[0])
        neutral_p = np.array(neutral_p)

        xx = np.average(neutral_p[:,0])
        yy = np.average(neutral_p[:,1])
        N[i].center_position = np.array([[xx, yy]])   #(1,2)
    return N
    
# 初始化位置
def position_initial_fixed_position(i): 
    if i == 0:
        return np.array([[1800,100]])
    if i == 1:
        return np.array([[1800,200]])
    if i == 2:
        return np.array([[1800,300]])
    if i == 3:
        return np.array([[1800,400]])
    if i == 4:
        return np.array([[1800,500]])
    if i == 5:
        return np.array([[1800,600]])
    if i == 6:
        return np.array([[1800,700]])
    if i == 7:
        return np.array([[1800,800]])

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
    
########

#%%#  一些预处理参数
M = Map()                       # 实例化地图
list_old = M.taboo_center.copy() # 判断列表是否发生变化
taboo_zone = taboo_zone_cal(M)

func = Fitness()                # 实例化适应度类
wind = Wind()                   # 实例化风向类
figure = Figure_plot()          # 实例化绘图类
obstacle = Obstacle()           # 实例化障碍物类
obstacles_info = obstacle.obstacle_define(M.obstacle_list) # 获取障碍物数组
fluent_python = Fluent2python() # 实例化fluent数据类
uds_t, grid_x, grid_y = fluent_python.file_to_array()      # 导入fluent数据


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
    P[i].position = position_initial_fixed_position(i) # 固定位置初始化
    # P[i].position = position_initial_random_position(obstacles_info, M.dim, M.map_x_lower, M.map_x_upper, M.map_y_lower, M.map_y_upper) # 随机位置初始化
    P[i].velocity = velocity_initial(M.dim, P[i].max_val, P[i].min_val) # 速度初始化

    # 记录
    P[i].fitness_history.append(P[i].fitness)
    P[i].position_history.append(P[i].position)    
# =============================================================================
    # 设置小生境成员
    # 0, 1, 2, 3 属第0个小生境
    # 4, 5, 6, 7 属第1个小生境
    if i == 0 or i == 1 or i == 2 or i == 3 :
        P[i].belonged_niche_NO = 0
    else:
        P[i].belonged_niche_NO = 1
        
    # 设置电荷属性, 骨干机器人不需要考虑电荷属性
    if i == 0 or i == 1 or i == 4 or i == 5:
        P[i].coulomb = 0
    else:
        P[i].coulomb = 1
# =============================================================================


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
    
#%% 开始循环
for iter_num in range(pn_class.total_iter):
    
    #---------------检测列表是否反生变化
    list_new = M.taboo_center # 判断列表是否发生变化
    if list_new != list_old:
        taboo_zone = taboo_zone_cal(M)
        list_old = list_new.copy()
    #--------------- 
    
    # 更新小生境信息
    for i in range(pn_class.niche_num):
        # 先清空一下
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
    # total_force = 1000000*repulsion_force_2(P, N, M) # 原始论文中的方法求斥力
    total_force = repulsion_force_path(P, N, M) # 利用路径方法求斥力

    # 更新粒子信息
    for i in range(pn_class.particle_num):
        
        P[i].force = total_force[i].reshape(1,2)
        # P[i].wind = wind.wind_func(P[i].position, wind_t)
        
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
        # if N[i].actived == True:
        #     # 三个判断条件
        #     if  N[i].agregation > N[i].epsilon_nm and N[i].gbest_fitness > 0.09: #N[i].C_threshold_nm:   #这里如果更换数据，相应阈值也要变化
        #         N[i].actived = False
        
        # 开始PSO算法
        if N[i].actived == True:
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
                # P[b].position = taboo_zone_avoid(P[b].position, P[b].velocity, M)     
                P[b].position = boundary_handle(P[b].position, M)
                # 躲避禁区
                P[i].position, P[i].velocity = taboo_zone_avoid_force(P[i].position, P[i].velocity, M, taboo_zone)
                P[b].position = boundary_handle(P[b].position, M)
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
                       
        # 开始NM算法
        # if N[i].actived == False:
        #     print("NM")
        #     nelder_mead(P, N[i], M)
        
        N[i].gbest_position_history.append(N[i].gbest_position)
        N[i].gbest_fitness_history.append(N[i].gbest_fitness)
    
    for j in range(pn_class.particle_num):
        # 记录
        P[j].fitness_history.append(P[j].fitness)
        P[j].position_history.append(P[j].position)
    
    # 设置搜索禁忌区
    for i in range(pn_class.niche_num):
        if N[i].actived == True:
            # 三个判断条件
            if  N[i].agregation > N[i].epsilon and N[i].gbest_fitness > N[i].C_threshold:
                print(f"污染源坐标为：{N[i].gbest_position}")
                M.taboo_center.append(N[i].gbest_position)

    # figure.niche_figure_plot(grid_x, grid_y, uds_t, obstacles_info, P, N)
    
figure.figure_plot(grid_x, grid_y, uds_t, obstacles_info, P, N)