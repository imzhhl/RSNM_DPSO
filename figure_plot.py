# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import re
import numpy as np
import pn_class
plt.ion()# 打开交互模式

matplotlib.rc("font", family='Microsoft YaHei')
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}


class Figure_plot:
    def __init__(self):
        pass
    def figure_plot_1(self, P, obstacles_info):
        plt.cla()
        # 画图, 障碍物和粒子初始点坐标    
        for i in range(len(P)):
            if P[i].belonged_niche_NO == 0:
                colorful = 'blue'
            if P[i].belonged_niche_NO == 1:
                colorful = 'red'
            plt.scatter(P[i].position[0][0],P[i].position[0][1], color = colorful)        
        plt.axis ([0, 371, 0, 393])
        plt.imshow(1-obstacles_info)
        plt.pause(0.5)
        
    def figure_plot(self, grid_x, grid_y, uds_t, obstacles_info, P, N):
        
        fig = plt.figure(figsize=(20,10))
        # 污染物浓度等值线图
        ax = fig.add_subplot(2,2,1)
        # 画图, 障碍物和粒子初始点坐标 
        for i in range(len(P)):

            if P[i].belonged_niche_NO == 0:
                colorful = 'blue'
                tag = 'member_1'
                mark = '.'
                ss = 50
            if P[i].belonged_niche_NO == 1:
                colorful = 'red'
                tag = 'member_2'
                mark = '.'
                ss = 50
            ax.scatter(P[i].position_history[0][0][0], P[i].position_history[0][0][1], marker = mark, s = ss, c = colorful, label = tag)           
        # plt.imshow(1-obstacles_info)
        # 污染物扩散二维等值线图
        matplotlib.pyplot.contour(grid_x, grid_y, uds_t[0,:,:],
                                  colors=list(["purple","blue","cyan", "green","yellow","orange","red"]),
                                  levels = [0.03, 0.04, 0.05, 0.06, 0.08, 0.17], linestyles=['-'])
        plt.axis ([0, 371, 0, 393])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("污染物浓度场和初始点分布")

        # 速度场流线图
        # ax = fig.add_subplot(2,2,2)
        # plt.pause(0.5)
        # # make a stream function:
        # # 确定wind_U和wind_V
        # wind_U = wind_t[:,:,0]
        # wind_V = wind_t[:,:,1]
        # ax.streamplot(grid_x, grid_y, wind_U, wind_V, density=0.8, linewidth=1.5, color="blue", arrowsize=1.5)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_title("流线图")  
        
        # 障碍物和粒子聚集结果, 不同小生境最佳值的迭代过程  
        ax = fig.add_subplot(2,2,3)
        for i in range(len(P)):
            
            if P[i].belonged_niche_NO == 0 and P[i].main == True:
                colorful = 'blue'
                tag = 'main_1'
                mark = '*'
                ss = 100
            if P[i].belonged_niche_NO == 1 and P[i].main == True:
                colorful = 'red'  
                tag = 'main_2'
                mark = '*'
                ss = 100
            if P[i].belonged_niche_NO == 0 and P[i].main == False:
                colorful = 'blue'
                tag = 'member_1'
                mark = '.'
                ss = 50
            if P[i].belonged_niche_NO == 1 and P[i].main == False:
                colorful = 'red'
                tag = 'member_2'
                mark = '.'
                ss = 50
            ax.scatter(P[i].position[0][0], P[i].position[0][1], marker = mark, s = ss, c = colorful, label = tag)  


        ax.plot(np.array(N[0].gbest_position_history)[:,0,0], np.array(N[0].gbest_position_history)[:,0,1], color="blue", label="niche_1")
        ax.plot(np.array(N[1].gbest_position_history)[:,0,0], np.array(N[1].gbest_position_history)[:,0,1], color="red", label="niche_2")
            
        matplotlib.pyplot.contour(grid_x, grid_y, uds_t[0,:,:],
                                  colors=list(["purple","blue","cyan", "green","yellow","orange","red"]),
                                  levels = [0.03, 0.04, 0.05, 0.06, 0.08, 0.17], linestyles=['-'])
        # ax.imshow(1-obstacles_info, 'gray')
        plt.xlim(0, 371)
        plt.ylim(0, 393)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title("UAVs搜索过程") 
        
        ax = fig.add_subplot(2,2,4)
        # PSO迭代过程浓度值变化
        ax.plot(N[0].gbest_fitness_history, color="blue", label="niche_1")
        ax.plot(N[1].gbest_fitness_history, color="red", label="niche_2")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.legend()
        ax.set_title("PSO迭代过程")  
        
    def niche_figure_plot(self, grid_x, grid_y, uds_t, obstacles_info, P, N, M):
        plt.cla()    
        # for i in range(len(P)):
            
        #     if P[i].belonged_niche_NO == 0:
        #         colorful = 'blue'
        #         tag = 'main_1'
        #         mark = '.'
        #         ss = 100
        #     if P[i].belonged_niche_NO == 1:
        #         colorful = 'red'  
        #         tag = 'main_2'
        #         mark = '*'
        #         ss = 100
        #     plt.scatter(P[i].position[0][0], P[i].position[0][1], marker = mark, s = ss, c = colorful, label = tag)  

        for i in range(pn_class.niche_num): # 未在PSO中
            for menber in N[i].menbers:
                b = int(re.findall("\d+",menber)[0])
                if N[i].in_PSO == False:
                    # plt.plot(np.array(P[b].position_history)[:,:,0], np.array(P[b].position_history)[:,:,1], color = 'green')
                    plt.scatter(P[b].position[0][0], P[b].position[0][1], marker = '.', s = 300, c = 'red', label = f"SINGLE_No. {b}")
         
        for i in range(pn_class.niche_num): # 在PSO中
            if N[i].in_PSO == True:
                plt.scatter(np.array(N[i].gbest_position_history)[-1][0][0], np.array(N[i].gbest_position_history)[-1][0][1], s = 300, marker = '*', label=f"niche_{i}")  
                # plt.plot(np.array(N[i].gbest_position_history)[:,0,0], np.array(N[i].gbest_position_history)[:,0,1] )
            
        
        matplotlib.pyplot.contour(grid_x, grid_y, uds_t[0,:,:],
                                  colors=list(["purple","blue","cyan", "green","orange","red"]),
                                  levels = [0.01, 0.04, 0.05, 0.06, 0.08, 0.17], linestyles=['-'])
        
        if len(M.taboo_center) != 0: # 如果已经形成了禁区
            for i in range(len(M.taboo_center)):
                circle = plt.Circle((M.taboo_center[i][0], M.taboo_center[i][1]),M.taboo_radius, edgecolor='black',facecolor='red', alpha = 0.5)
                plt.gca().add_patch(circle)      
        
        plt.imshow(obstacles_info,'Greys')
        plt.axis ([0, M.map_x_upper, 0, M.map_y_upper])
        # ax.imshow(1-obstacles_info, 'gray')
        plt.xlim(0, 371)
        plt.ylim(0, 393)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("UAVs搜索过程") 
        plt.pause(0.01)
        plt.ioff()