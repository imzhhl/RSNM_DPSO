# -*- coding: utf-8 -*-
from numpy import array
global total_iter, particle_num, niche_num

# ---------------------------------------------------------------
#total_iter = 300              # 迭代次数
#particle_num = 16             # 初始化粒子群总个体数量
niche_num = 2                 # 小生境的数量
#response_time = 1             # 响应时间
#errorness = 0.01              # 传感器误差
# ---------------------------------------------------------------


class Map:
    def __init__(self):
        self.dim = 2
        self.C_threshold = 0.01           #收敛进入羽流阈值
        self.map_x_lower = 0              # 计算域x下限
        self.map_x_upper = 371           # 计算域x上限
        self.map_y_lower = 0              # 计算域y下限
        self.map_y_upper = 393           # 计算域y上限
        self.taboo_radius = 10           # 禁忌区半径
        self.taboo_center = []            # 初始化禁忌区的位置

        # 障碍物位置列表
        self.dx = 1.21
        self.dy = 3.95
        self.obstacle_list = array([
                                    [101.243, 106.1],
                                    [116.563, 106.1],
                                    [131.383, 106.1],
                                    [147.503, 106.1],
                                    [162.523, 106.1],
                                    [177.543, 106.1],
                                    [193.463, 106.1],
                                    [208.483, 106.1],
                                    [223.503, 106.1],
                                    [239.423, 106.1],
                                    [254.443, 106.1],
                                    [269.763, 106.1],
                                    [101.243, 126.2],
                                    [116.563, 126.2],
                                    [131.383, 126.2],
                                    [147.503, 126.2],
                                    [162.523, 126.2],
                                    [177.543, 126.2],
                                    [193.463, 126.2],
                                    [208.483, 126.2],
                                    [223.503, 126.2],
                                    [239.423, 126.2],
                                    [254.443, 126.2],
                                    [269.763, 126.2],
                                    [101.243, 146.3],
                                    [116.563, 146.3],
                                    [131.383, 146.3],
                                    [147.503, 146.3],
                                    [162.523, 146.3],
                                    [177.543, 146.3],
                                    [193.463, 146.3],
                                    [208.483, 146.3],
                                    [223.503, 146.3],
                                    [239.423, 146.3],
                                    [254.443, 146.3],
                                    [269.763, 146.3],
                                    [101.243, 166.4],
                                    [116.563, 166.4],
                                    [131.383, 166.4],
                                    [147.503, 166.4],
                                    [162.523, 166.4],
                                    [177.543, 166.4],
                                    [193.463, 166.4],
                                    [208.483, 166.4],
                                    [223.503, 166.4],
                                    [239.423, 166.4],
                                    [254.443, 166.4],
                                    [269.763, 166.4],
                                    [101.243, 186.5],
                                    [116.563, 186.5],
                                    [131.383, 186.5],
                                    [147.503, 186.5],
                                    [162.523, 186.5],
                                    [177.543, 186.5],
                                    [193.463, 186.5],
                                    [208.483, 186.5],
                                    [223.503, 186.5],
                                    [239.423, 186.5],
                                    [254.443, 186.5],
                                    [269.763, 186.5],
                                    [101.243, 206.6],
                                    [116.563, 206.6],
                                    [131.383, 206.6],
                                    [147.503, 206.6],
                                    [162.523, 206.6],
                                    [177.543, 206.6],
                                    [193.463, 206.6],
                                    [208.483, 206.6],
                                    [223.503, 206.6],
                                    [239.423, 206.6],
                                    [254.443, 206.6],
                                    [269.763, 206.6],
                                    [101.243, 226.7],
                                    [116.563, 226.7],
                                    [131.383, 226.7],
                                    [147.503, 226.7],
                                    [162.523, 226.7],
                                    [177.543, 226.7],
                                    [193.463, 226.7],
                                    [208.483, 226.7],
                                    [223.503, 226.7],
                                    [239.423, 226.7],
                                    [254.443, 226.7],
                                    [269.763, 226.7],
                                    [101.243, 246.8],
                                    [116.563, 246.8],
                                    [131.383, 246.8],
                                    [147.503, 246.8],
                                    [162.523, 246.8],
                                    [177.543, 246.8],
                                    [193.463, 246.8],
                                    [208.483, 246.8],
                                    [223.503, 246.8],
                                    [239.423, 246.8],
                                    [254.443, 246.8],
                                    [269.763, 246.8],
                                    [101.243, 266.9],
                                    [116.563, 266.9],
                                    [131.383, 266.9],
                                    [147.503, 266.9],
                                    [162.523, 266.9],
                                    [177.543, 266.9],
                                    [193.463, 266.9],
                                    [208.483, 266.9],
                                    [223.503, 266.9],
                                    [239.423, 266.9],
                                    [254.443, 266.9],
                                    [269.763, 266.9],
                                    [101.243, 287.3],
                                    [116.563, 287.3],
                                    [131.383, 287.3],
                                    [147.503, 287.3],
                                    [162.523, 287.3],
                                    [177.543, 287.3],
                                    [193.463, 287.3],
                                    [208.483, 287.3],
                                    [223.503, 287.3],
                                    [239.423, 287.3],
                                    [254.443, 287.3],
                                    [269.763, 287.3]])


class Particle:
    def __init__(self):
        self.NO = float('nan') # 编号
        self.belonged_niche_NO = float('nan') # 所属小生境
        self.in_plume = False
        self.step = 5
        self.position = array([[float('-inf'),float('-inf')]]) # 初始化粒子的位置
        self.position_history = []   
        
        self.fitness = float('nan') # 初始化粒子的适应度
        self.fitness_history = []
        
        self.velocity = array([[float('-inf'),float('-inf')]]) # 初始化粒子的速度, 二维+
        self.unit_vector = array([[float('-inf'),float('-inf')]])
        self.wind = array([[float('-inf'),float('-inf')]])      # 初始化风速信息
        
        self.force = array([[0,0]])  # 粒子所受力
        
        self.pbest_fitness  = float('-inf')  # 粒子的最佳适应度
        self.pbest_position = array([[float('-inf'),float('-inf')]]) # 粒子的最佳位置
          
        self.max_val =  20 # 速度限制
        self.min_val = -self.max_val
        
        self.c_ini_1 = 2.5
        self.c_end_1 = 0.5
        self.c_ini_2 = 0.5
        self.c_end_2 = 2.5
        
        self.w = 0.9 # 设置惯性权重
        self.c1 = 2  # 设置个体学习系数
        self.c2 = 2  # 设置全局学习系数
        self.c3 = 0.7# 设置个体反向学习系数
        self.c4 = 0.3# 设置全局反向学习系数
        
        #self.r_attract = 0  # 骨干粒子捕获半径
        
        

class Niche:
    def __init__(self):
        self.NO = float('nan') # 编号
                
        self.UAVs_position = {}
        self.UAVs_fitness = {}
        self.UAVs_unit_vector = {}
        
        self.in_PSO = False
        self.in_NM = False
        self.menbers = []      # 小生境成员, 通过len(NICHE[i].menbers) != 0判断小生境是否被激活
        
        self.center_position = array([[float('-inf'),float('-inf')]])    # 小生境中心位置
        self.center_position_history = [] # 小生境中心位置历史
        
        self.force = array([[0,0]])  # 小生境受到的力
        #self.r_repulse = 20          # 小生境的排斥半径
        
        self.gbest_fitness = float('-inf')   # 种群的最佳适应度
        self.gbest_fitness_history = [] # 种群最佳适应度的变化情况
        self.gbest_position = array([[float('-inf'),float('-inf')]])    # 最佳粒子的位置
        self.gbest_position_history = [] # 种群最佳粒子的历史位置
          
        self.agregation = float('nan')  # 小生境的聚集度
        #self.niche_Gr = self.r_repulse  # 用于计算聚集度的小生境半径
        
        self.C_threshold_end = 0.08 # 浓度度收敛阈值
        self.epsilon_end = 0.4      # 聚集度收敛阈值
    
        self.C_threshold_nm = 0.05 # 进入NM是阈值
        self.epsilon_nm = 0.2      # 进入NM算法的聚集度阈值
        