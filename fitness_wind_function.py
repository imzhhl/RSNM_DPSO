# -*- coding: utf-8 -*-
import math
import numpy as np


class Fitness:
    def __init__(self, **kwargs):
        self.x_limit = kwargs.pop('x_limit', 371)
        self.y_limit = kwargs.pop('y_limit', 393)
    
    # 求每个点的适应度函数
    def fitness_func(self, X, uds_t):
        x = X[:, 0] # 粒子x坐标
        y = X[:, 1] # 粒子y坐标
        a = math.floor(x) # 对x坐标取整
        b = math.floor(y) # 对y坐标取整
        
        # 防止索引超标
        if a >=  self.x_limit - 1 or b >= self.y_limit - 1:
            a = self.x_limit - 2
            b = self.y_limit - 2
        if a < 0 or b < 0:
            a = 0
            b = 0
            
        #二维线性插值, 先检索y(行), 再检索x(列)
        z1 = uds_t[0, b,   a] + (uds_t[0, b,   a+1] - uds_t[0, b,   a]) * (x - a)
        z2 = uds_t[0, b+1, a] + (uds_t[0, b+1, a+1] - uds_t[0, b+1, a]) * (x - a)
        z = z1 + (z2 - z1) * (y - b)
        
        return z  # 浮点数, 单个粒子位置X处的污染浓度值

class Wind:    
    def __init__(self, **kwargs):
        self.x_limit = kwargs.pop('x_limit', 371)
        self.y_limit = kwargs.pop('y_limit', 393)

    # 求每个点的风向（二维数组）
    def wind_func(self, X, wind_t):
        x = X[:, 0]
        y = X[:, 1]
        #二维线性插值
        z = np.zeros((len(x),2)) #风向, 二维数组 
        for i in range(len(x)):
            a = math.floor(x[i])
            b = math.floor(y[i])
            
            # 防止索引超标
            if a >= self.x_limit - 1 or b >=  self.y_limit - 1:
                a = self.x_limit - 2
                b = self.y_limit - 2
            if a < 0 or b < 0:
                a = 0
                b = 0
                
            z1 = wind_t[b, a, :] + (wind_t[b, a+1, :]-wind_t[b, a, :]) * (x[i] - a)
            z2 = wind_t[b+1, a, :] + (wind_t[b+1, a+1, :]-wind_t[b+1, a, :]) * (x[i] - a)
            z[i] = z1 + (z2 - z1) * (y[i] - b)
        return z  #二维数组，保存了各粒子位置插值得到的U和V速度
    
    # wind utilization controled by parameter "cp" (chi_theta)
    def cp(self, X, wind, V):  #控制参数
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