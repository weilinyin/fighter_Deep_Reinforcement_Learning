import numpy as np
from math import sin , cos , tan
from collections import deque
import gymnasium as gym
import copy

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
        # 初始化相对量

        self.r = np.abs(target.position - chaser.position)
        self.q_y = np.arcsin((target.position[1] - chaser.position[1]) / self.r)
        self.q_z = np.arccos((target.position[0] - chaser.position[0]) / (np.abs(target.position[0:1] - chaser.position[0:1])))
        self.chaser = chaser
        self.target = target
        self.theta = self.chaser.theta
        self.psi = self.chaser.psi

    def check_detection(self):
        # 检测战机是否探测到导弹
        if self.r < self.chaser.R:
            return True
        else:
            return False



    def state(self):
        # 输出相对状态量

        r = self.r
        q_y = self.q_y
        q_z = self.q_z

       # 归一化处理

        r_bar = r / self.chaser.R
        eta_y = np.arcsin(-sin(self.theta) * cos(self.psi) * cos(q_y) * cos(q_z) +
                          cos(self.theta) * sin(q_y) -
                          sin(self.theta) * cos(self.psi) * cos(q_y) * sin(q_z))
        eta_z = np.arctan((cos(q_y) * sin(q_z) * cos(self.psi) -cos(q_y) * cos(q_z) * sin(self.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.theta) * cos(self.psi) +
                           sin(self.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.theta) * sin(self.psi)))
        return r_bar , eta_y , eta_z




    
    def proportional_navigation(self):
        # 比例导引法

        self.dtheta = 3 * self.dq_y * cos(self.q_z - self.chaser.psi)
        self.dpsi = 3 * self.dq_z - 3 * self.dq_y * tan(self.chaser.theta) * sin(self.q_z - self.chaser.psi)



    def simulate(self,dt):
        # 模拟飞行器运动

        self.dr = (self.target.velocity * (cos(self.target.theta) * cos(self.q_y) * cos(self.target.psi - self.q_z) +
                                            sin(self.target.theta) * sin(self.q_y)) -
                    self.chaser.velocity * (cos(self.theta) * cos(self.q_y) * cos(self.psi - self.q_z) +
                                            sin(self.theta) * sin(self.q_y)))
            
        self.dq_y = (self.target.velocity * (sin(self.target.theta) * cos(self.q_y) -
                                            cos(self.target.theta) * sin(self.q_y) * cos(self.target.psi - self.q_z) +
                                            sin(self.target.psi - self.q_z) * sin(self.q_y)) -
                        self.chaser.velocity * (sin(self.theta) * cos(self.q_y) -
                                            cos(self.theta) * sin(self.q_y) * cos(self.psi - self.q_z) +
                                            sin(self.psi - self.q_z) * sin(self.q_y)))/self.r
        
        self.dq_z = (self.target.velocity * cos(self.target.theta) * sin(self.psi - self.q_z) -
                        self.chaser.velocity * cos(self.theta) * sin(self.psi - self.q_z))/(self.r * cos(self.q_y))
        

        self.r += self.dr * dt
        self.q_y += self.dq_y * dt
        self.q_z += self.dq_z * dt

        self.theta += self.dtheta * dt 
        self.psi += self.dpsi * dt



        #self.chaser.update(self.dtheta , self.dpsi , dt)
    










    


# 1. 环境模块
class FighterEnv(gym.Env):

    def __init__(self):
        
        self.observation_space = gym.spaces.Box(low = np.array([0 , -1 , -1 , -np.inf , -1 , -1]) ,
                                                high = np.array([np.inf , 1 , 1 , 2 , 1 , 1]) ,
                                                shape= (6,),
                                                dtype = np.float64)
        
        self.action_space = gym.spaces.Box(low = np.array([-np.pi , 0]) ,
                                           high = np.array([np.pi , np.sqrt(2)]) ,
                                           shape = (2,),
                                           dtype = np.float64) #第一项为加速度角度，第二项为模长
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


        # 规定时间步长
        self.dt = DT

        # 先进行比例导引法
        self.start_simulate()


        # 初始化状态
        r_1 , q_y1 , q_z1 = self.FD.state()
        r_2 , q_y2 , q_z2 = self.FT.state()

        self.state = np.array([r_1 , q_y1 , q_z1 , r_2 , q_y2 , q_z2])

        # 存档
        self.saves = {"figher":copy.deepcopy(self.fighter),"defender":copy.deepcopy(self.defender),"state":copy.deepcopy(self.state)}


    def start_simulate(self):

        while not self.FD.check_detection():
            self.FD.proportional_navigation()
            self.FD.simulate(DT)
            self.FT.proportional_navigation()
            self.FT.simulate(DT)



    def reset(self , seed=None, options=None):
        # 重置环境，返回初始状态
        # 初始化战斗机、防御弹和目标参数（速度、位置、制导律等）
        self.fighter = copy.deepcopy(self.saves["figher"])
        self.defender = copy.deepcopy(self.saves["defender"])
        self.state = copy.deepcopy(self.saves["state"])

        observation = self.state
        return observation , {}



    def step(self, action):
        # 根据动作更新状态，返回新状态、奖励、终止标志
        # 实现运动方程和防御弹逻辑
    
        # 按时间步长模拟飞行器运动

        # 防御弹
        self.FD.proportional_navigation()
        self.FD.simulate(DT)

        # 战斗机
        a_y = action[1] * cos(action[0])
        a_z = action[1] * sin(action[0]) # 根据动作计算加速度

        self.FT.dtheta = a_y/self.fighter.velocity
        self.FT.dpsi = -a_z/(self.fighter.velocity * cos(self.fighter.theta))

        self.FT.simulate(DT)  # 更新战斗机状态



        

        r_1 , q_y1 , q_z1 = self.FD.state()
        r_2 , q_y2 , q_z2 = self.FT.state()

        observation = np.array([r_1 , q_y1 , q_z1 , r_2 , q_y2 , q_z2])

        reward = self.calculate_reward()
        terminated = self.check_terminated(observation)
        truncated = False





        return observation, reward, terminated, truncated , {}
    


    def check_terminated(self,state):
        # 检查是否终止，如飞行器超出范围或被拦截
        if state[3] * R_FD <= R_DAM:
            return True # 被拦截
        else:
           return False # 未被拦截






    
    def calculate_reward(self):
        # 根据论文公式计算奖励（突防、被拦截、控制能量等）
        if self.state[3] <= 0 and self.FD.dr > 0 and self.FD.r > R_DAM and :

            reward = 
