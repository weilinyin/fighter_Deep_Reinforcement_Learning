import numpy as np
from math import sin , cos , tan
from collections import deque

R_DAM = 20
R_FD = 18000
R_FT = 30000
ETA_FT = 10 * np.pi / 180
DT = 0.1


class aircraft:
    def __init__(self,position,theta,psi,velocity,a_max,R):
        self.position = position # 位置 (x, y, z)
        self.theta = theta # 弹道倾角 (theta)
        self.psi = psi # 弹道偏角 (psi)
        self.velocity = velocity # 速度 (v)
        self.a_max=a_max # 最大过载 (a_max)
        self.R = R # r*
    

    def update(self,dtheta,dpsi,dt):
        # 更新状态
        self.position += np.array([self.velocity * cos(self.theta) * sin(self.psi),
                                     self.velocity * sin(self.theta),
                                    -self.velocity * cos(self.theta) * sin(self.psi)]) * dt
        self.theta += dtheta * dt 
        self.psi += dpsi * dt 




class relative:
    def __init__(self,chaser:aircraft,target:aircraft):
        #初始化相对量

        self.r = np.abs(target.position - chaser.position)
        self.q_y = np.asin((target.position[1] - chaser.position[1]) / self.r)
        self.q_z = np.acos((target.position[0] - chaser.position[0]) / (np.abs(target.position[0:1] - chaser.position[0:1])))
        self.chaser = chaser
        self.target = target
        self.theta = self.chaser.theta
        self.psi = self.chaser.psi

    def state(self):
        #输出相对状态量

        r = np.abs(self.target.position - self.chaser.position)
        q_y = np.asin((self.target.position[1] - self.chaser.position[1]) / r)
        q_z = np.acos((self.target.position[0] - self.chaser.position[0]) / (np.abs(self.target.position[0:1] - self.chaser.position[0:1])))

        r_bar = r / self.chaser.R
        eta_y = np.arcsin(-sin(self.theta) * cos(self.psi) * cos(q_y) * cos(q_z) +
                          cos(self.theta) * sin(q_y) -
                          sin(self.theta) * cos(self.psi) * cos(q_y) * sin(q_z))
        eta_z = np.arctan((cos(q_y) * sin(q_z) * cos(self.psi) -cos(q_y) * cos(q_z) * sin(self.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.theta) * cos(self.psi) +
                           sin(self.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.theta) * sin(self.psi)))
        return r_bar , eta_y , eta_z




    
    def simulate(self,dt,a_y,a_z,Isaction:bool):

        if Isaction == False:
            #比例导引法
            self.dr = (self.target.velocity * (cos(self.target.theta) * cos(self.q_y) * cos(self.target.psi - self.q_z) +
                                                sin(self.target.theta) * sin(self.q_y)) -
                        self.chaser.velocity * (cos(self.chaser.theta) * cos(self.q_y) * cos(self.chaser.psi - self.q_z) +
                                                sin(self.chaser.theta) * sin(self.q_y)))
            
            self.dq_y = (self.target.velocity * (sin(self.target.theta) * cos(self.q_y) -
                                                cos(self.target.theta) * sin(self.q_y) * cos(self.target.psi - self.q_z) +
                                                sin(self.target.psi - self.q_z) * sin(self.q_y)) -
                        self.chaser.velocity * (sin(self.chaser.theta) * cos(self.q_y) -
                                                cos(self.chaser.theta) * sin(self.q_y) * cos(self.chaser.psi - self.q_z) +
                                                sin(self.chaser.psi - self.q_z) * sin(self.q_y)))/self.r
            
            self.dq_z = (self.target.velocity * cos(self.target.theta) * sin(self.chaser.psi - self.q_z) -
                        self.chaser.velocity * cos(self.chaser.theta) * sin(self.chaser.psi - self.q_z))/(self.r * cos(self.q_y))
            

            self.dtheta = 3 * self.dq_y * cos(self.q_z - self.chaser.psi)
            self.dpsi = 3 * self.dq_z - 3 * self.dq_y * tan(self.chaser.theta) * sin(self.q_z - self.chaser.psi)

            self.r += self.dr * dt
            self.q_y += self.dq_y * dt
            self.q_z += self.dq_z * dt

        else:

            self.dtheta = a_y/self.chaser.velocity
            self.dpsi = -a_z/(self.chaser.velocity * cos(self.chaser.theta))
        
        
        self.chaser.update(self.dtheta , self.dpsi , dt)
        


    


# 1. 环境模块
class FighterEnv:
    def __init__(self):

        self.reset()

    def reset(self):
        # 重置环境，返回初始状态
        # 初始化战斗机、防御弹和目标参数（速度、位置、制导律等）
        self.fighter = aircraft(
            np.array([0, 10000, 0]),
            0,
            0,
            900,
            2,
            18000
        )
        self.defender = aircraft(
            np.array([40000, 10000, -5000]),
            0,
            3.23,
            1000,
            5,
            30000
        )
        self.target = aircraft(
            np.array([50000,10000,0]),
            0,
            0,
            0,
            0,
            0
        )


        # 初始化相对量
        self.FD = relative(self.defender, self.target)
        self.FT = relative(self.fighter, self.target)

        # 初始化状态
        r_1 , q_y1 , q_z1 = self.FD.state()
        r_2 , q_y2 , q_z2 = self.FT.state()

        self.state = np.array([r_1 , q_y1 , q_z1 , r_2 , q_y2 , q_z2])

        # 规定时间步长
        self.dt = DT


    def step(self, action):
        # 根据动作更新状态，返回新状态、奖励、终止标志
        # 实现运动方程和防御弹逻辑
    
        # 按时间步长模拟飞行器运动
        self.FD.simulate(self.dt , action[0] , action[1], self.state[0] <= R_FD)
        self.FT.simulate(self.dt , 0 , 0 , False)

        r_1 , q_y1 , q_z1 = self.FD.state()
        r_2 , q_y2 , q_z2 = self.FT.state()

        next_state = np.array([r_1 , q_y1 , q_z1 , r_2 , q_y2 , q_z2])

        reward = self.calculate_reward()
        done = self.check_done(next_state)




        return next_state, reward, done
    


    def check_done(self,state):
        # 检查是否终止，如飞行器超出范围或被拦截
        if state[3] * R_FD <= R_DAM:
            return True # 被拦截
        else 
           return False # 未被拦截






    
    def calculate_reward(self):
        # 根据论文公式计算奖励（突防、被拦截、控制能量等）
        
        return reward