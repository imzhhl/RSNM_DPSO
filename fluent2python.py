import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib

from scipy.interpolate import griddata



project_root = os.path.dirname(os.path.realpath(__file__))


class Fluent2python:
    """
    
    从fluent中导出的数据导入到numpy数组中
    
    """
    def __init__(self):
        self.path = os.path.join(project_root, "")
        self.fluent_file_path = os.path.join(project_root, "fluent111")
        self.dispersed_x = 2000 # x的离散数量
        self.dispersed_y = 1000 # y的离散数量
        
    def file_to_array(self):
        #x,y的坐标
        x=[]; y=[]
        #创建uds为一维数组
        uds = np.zeros(1)
        #创建W为二维数组
        wind = np.zeros((1,2)) 
        #创建两个列表
        uds_t = []  #浓度
        wind_t = []  #风向
        
        with open(self.fluent_file_path,'r')  as  csvfile:   
            #指定分隔符为","
            plots=csv.reader(csvfile,delimiter=',')
            #循环读取文件各列
            for row in plots:
                #跳过文件前面的非数据行  
                if plots.line_num == 1:
                    continue
                # 读取x和y坐标和风向
                x.append(float(row[1]))
                y.append(float(row[2]))
                #读取风向
                # wind = np.append(wind,[[float(row[3]),float(row[4])]],axis=0)
                #读取浓度
                uds = np.append(uds,float(row[3]))
        
        #删掉第0个元素
        # wind = np.delete(wind,0,axis=0)
        uds = np.delete(uds,0)
      
        # 如果数值过小，则视为0，防止出现非数的情况
        for i in range(len(y)):
            if uds[i] < 1e-10:
                uds[i] =0.0
        
        # 确定meshgrid的坐标范围，并离散坐标        
        xi=np.linspace(min(x),max(x), self.dispersed_x)
        yi=np.linspace(min(y),max(y), self.dispersed_y)
        
        # grid_x,grid_y坐标必须维数一致，且为二维
        global grid_x, grid_y
        grid_x, grid_y = np.meshgrid(xi, yi)
        
        # griddata插值，并存入uds_t列表中，列表中的元素为二维数组
        uds_t.append(griddata((x,y), uds, (grid_x,grid_y), method='linear', fill_value = 0, rescale = False))
        # griddata插值，并存入wind_t列表中，列表中的元素为三维数组
        # wind_t.append(griddata((x,y), wind, (grid_x,grid_y), method='linear', fill_value = 0, rescale = False))
        
        # 将uds_t转换为三维np数组
        uds_t = np.array(uds_t)
        
        # 将wind_t转换为三维np数组
        # wind_t = np.array(wind_t)
        # wind_t = np.reshape(wind_t,(self.dispersed_y, self.dispersed_x,2))
        
        # 保存到文件中，下次使用是直接读取文件，避免重复上述过程(太慢！)
        np.save('uds_t.npy', uds_t)
        # np.save('wind.npy', wind_t)
        np.save('grid_x.npy', grid_x)
        np.save('grid_y.npy', grid_y)
        return (uds_t, grid_x, grid_y)
if __name__ == '__main__':
    fluent2python = Fluent2python()  # 函数实例化
    uds_t, grid_x, grid_y = fluent2python.file_to_array()   # 运行一次就够了
    matplotlib.pyplot.contour(grid_x, grid_y, uds_t[0,:,:],
                              colors=list(["purple","blue","cyan", "green","black","orange","red"]),
                              levels = [0.03, 0.04, 0.05, 0.06, 0.08, 0.17], linestyles=['-'])
    plt.xlim(0, 2000)
    plt.ylim(0, 1000)