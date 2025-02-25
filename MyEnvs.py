import numpy as np
from math import sin , cos , tan ,acos ,asin ,atan ,sqrt , tanh
from collections import deque
import gymnasium as gym
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


R_DAM = 20
R_FD = 18000
R_FT = 30000
ETA_FT = 10 * np.pi / 180
DT = 0.1
K_PLUS = 100
K_MINUS = -100
K_R1 = 0.1
K_R2 = 1
K_R3 = 2e-5




class aircraft:
    def __init__(self,position,theta:np.float64,psi:np.float64,velocity:np.float64,a_max:np.float64,R:np.float64):
        self.position = position # 位置 (x, y, z)
        self.theta = theta # 弹道倾角 (theta)
        self.psi = psi # 弹道偏角 (psi)
        self.velocity = velocity # 速度 (v)
        self.a_max=a_max # 最大过载 (a_max)
        self.R = R # r*
        self.a_y = 0
        self.a_z = 0

    def update(self, dt, a_y ,a_z):

        self.a_y = a_y
        self.a_z = a_z

        dposition = self.velocity * np.array([cos(self.theta) * cos(self.psi) , 
                                              sin(self.theta) , 
                                              -cos(self.theta) * sin(self.psi)])
        dtheta = a_y / self.velocity
        dpsi = - a_z / (self.velocity * cos(self.theta))

        self.position += dposition * dt
        self.theta += dtheta * dt
        self.psi +=dpsi * dt
        



class relative:
    def __init__(self,chaser:aircraft,target:aircraft):
        # 初始化相对量

        self.r = np.linalg.norm(target.position - chaser.position)
        self.r_0 = self.r
        self.r = np.array([self.r , self.r]) #[上一dt和这一dt]



        self.q_y = asin((target.position[1] - chaser.position[1]) / self.r[1])
        self.q_y = np.array([self.q_y , self.q_y])


        self.q_z = acos((target.position[0] - chaser.position[0]) / (np.linalg.norm(np.array([target.position[0] - chaser.position[0] , target.position[2] - chaser.position[2]])))) 
        self.q_z = np.array([self.q_z , self.q_z])


        self.dq_y = 0
        self.dq_z = 0
        self.dr = 0
        self.dq_y = 0
        self.dq_z = 0


        self.chaser = chaser
        self.target = target


    def check_detection(self):
        # 检测战机是否探测到导弹
        if self.r[1] < self.chaser.R:
            return True
        else:
            return False



    def state(self,type):
        # 输出相对状态量

        r = self.r[1]
        q_y = self.q_y[1]
        q_z = self.q_z[1]

       # 归一化处理
        if type == "defender":
            r_bar = r / self.chaser.R
            eta_y = (-sin(self.target.theta) * cos(self.target.psi) * cos(q_y) * cos(q_z) +
                          cos(self.target.theta) * sin(q_y) -
                          sin(self.target.theta) * cos(self.target.psi) * cos(q_y) * sin(q_z))
            eta_y = np.clip(eta_y, -1, 1)
            eta_y = asin(eta_y)
            eta_z = atan((cos(q_y) * sin(q_z) * cos(self.chaser.psi) -cos(q_y) * cos(q_z) * sin(self.chaser.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.chaser.theta) * cos(self.chaser.psi) +
                           sin(self.chaser.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.chaser.theta) * sin(self.chaser.psi)))
        elif type == "fighter":
            r_bar = (r-self.chaser.R) / (self.r_0 - self.chaser.R) 

            eta_y = (-sin(self.chaser.theta) * cos(self.chaser.psi) * cos(q_y) * cos(q_z) +
                          cos(self.chaser.theta) * sin(q_y) -
                          sin(self.chaser.theta) * cos(self.chaser.psi) * cos(q_y) * sin(q_z))
            eta_y = np.clip(eta_y, -1, 1)
            eta_z = atan((cos(q_y) * sin(q_z) * cos(self.chaser.psi) -cos(q_y) * cos(q_z) * sin(self.chaser.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.chaser.theta) * cos(self.chaser.psi) +
                           sin(self.chaser.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.chaser.theta) * sin(self.chaser.psi)))
        eta_y = tanh(eta_y)
        eta_z = tanh(eta_z)
        return r_bar , eta_y , eta_z




    
    def proportional_navigation(self):
        # 比例导引法

        dtheta = 3 * self.dq_y * cos(self.q_z[1] - self.chaser.psi)
        dpsi = 3 * self.dq_z - 3 * self.dq_y * tan(self.chaser.theta) * sin(self.q_z[1] - self.chaser.psi)

        a_y , a_z = self.calculate_a(dtheta,dpsi)

        return a_y , a_z


    
    def calculate_a(self,dtheta,dpsi):

        a_y = dtheta * self.chaser.velocity
        a_z = -dpsi * self.chaser.velocity * cos(self.chaser.theta)

        a = sqrt(a_y**2 + a_z**2)

        if a >self.chaser.a_max:
            a_y = a_y * self.chaser.a_max / a
            a_z = a_z * self.chaser.a_max / a


        return a_y , a_z


    def update(self):
        # 模拟过程中更新相对量

        # 更新微分量
        self.dr = self.r[1]-self.r[0]
        self.dq_y = self.q_y[1]-self.q_y[0]
        self.dq_z = self.q_z[1]-self.q_z[0]

        self.r[0] = self.r[1]
        self.q_y[0] = self.q_y[1]
        self.q_z[0] = self.q_z[1]

        self.r[1] = np.linalg.norm(self.target.position - self.chaser.position)
        self.q_y[1] = asin((self.target.position[1] - self.chaser.position[1]) / self.r[0])
        self.q_z[1] = acos((self.target.position[0] - self.chaser.position[0]) / 
                                      (np.linalg.norm(np.array([self.target.position[0] - self.chaser.position[0] , self.target.position[2] - self.chaser.position[2]])))) 









    


# 1. 环境模块
class FighterEnv(gym.Env):

    def __init__(self,Isprint = False):
        super().__init__()

        self.Isprint = Isprint

        
        self.observation_space = gym.spaces.Box(low = np.array([0 , -1 , -1 , -np.inf , -1 , -1]) ,
                                                high = np.array([1.1 , 1 , 1 , 2 , 2 , 1]) ,
                                                shape= (6,),
                                                dtype = np.float64)
        
        self.action_space = gym.spaces.Box(low = np.array([-1 , -1]) ,
                                           high = np.array([1 , 1]) ,
                                           shape = (2,),
                                           dtype = np.float64) #第一项为加速度角度，第二项为加速度大小
        self.fighter = aircraft(
            np.array([0, 10000, 0]).astype(np.float64),
            0,
            0,
            900,
            2*9.81,
            18000
        )
        self.defender = aircraft(
            np.array([40000, 5000, -5000]).astype(np.float64),
            0.11,
            3.23,
            1000,
            5*9.81,
            30000
        )
        self.target = aircraft(
            np.array([50000,0,0]).astype(np.float64),
            0,
            0,
            0,
            0,
            0
        )


        # 初始化相对量
        self.FD = relative(self.defender, self.fighter)
        self.FT = relative(self.fighter, self.target)


        # 规定时间步长
        self.dt = DT

        # 制图数据
        self.plotdata = {"fighter":{} , "defender":{}, "rewards":[] , "eta":[]} 

        self.plotdata["fighter"] = {"x":[], "y":[] , "z":[], "theta":[], "psi":[], "a_y":[], "a_z":[] , "r":[]}
        self.plotdata["defender"] = {"x":[], "y":[] , "z":[], "theta":[], "psi":[], "a_y":[], "a_z":[] , "r":[]} 

        # 计时器
        self.t = 0.0
        self.t_array = []

        # 未开始突防
        self.start_simulate()

        self.t_0=self.t



        # 初始化状态
        r_1 , q_y1 , q_z1 = self.FD.state("defender")
        r_2 , q_y2 , q_z2 = self.FT.state("fighter")


        self.state = np.array([r_1 , q_y1 , q_z1 , r_2 , q_y2 , q_z2])
        # 存档
        self.saves = {"figher":copy.deepcopy(self.fighter),"defender":copy.deepcopy(self.defender),"state":copy.deepcopy(self.state)}

        # 初始化成功突防标志
        self.success = False



        # 初始化失败标志
        self.fail = False





    def start_simulate(self):

        while not self.FD.check_detection():
            
            
            a_y , a_z = self.FT.proportional_navigation()
            self.fighter.update(self.dt, a_y , a_z)

            a_y , a_z = self.FD.proportional_navigation()
            self.defender.update(self.dt, a_y , a_z)

            self.FT.update()
            self.FD.update()
 
            # 加入观察数据
            if self.Isprint:
                self.calculate_eta()
                self.update_plotdata()



    def update_plotdata(self):
        self.plotdata["fighter"]["x"].append(self.fighter.position[0])
        self.plotdata["fighter"]["y"].append(self.fighter.position[1])
        self.plotdata["fighter"]["z"].append(self.fighter.position[2])
        self.plotdata["fighter"]["theta"].append(self.fighter.theta)
        self.plotdata["fighter"]["psi"].append(self.fighter.psi)
        self.plotdata["fighter"]["a_y"].append(self.fighter.a_y)
        self.plotdata["fighter"]["a_z"].append(self.fighter.a_z)
        self.plotdata["fighter"]["r"].append(self.FT.r)

        self.plotdata["defender"]["x"].append(self.defender.position[0])
        self.plotdata["defender"]["y"].append(self.defender.position[1])
        self.plotdata["defender"]["z"].append(self.defender.position[2])
        self.plotdata["defender"]["theta"].append(self.defender.theta)
        self.plotdata["defender"]["psi"].append(self.defender.psi)
        self.plotdata["defender"]["a_y"].append(self.defender.a_y)
        self.plotdata["defender"]["a_z"].append(self.defender.a_z)
        self.plotdata["defender"]["r"].append(self.FD.r)

        self.plotdata["eta"].append(self.eta_ft)

        self.t_array.append(self.t)
        self.t += self.dt




    def reset(self , seed=None, options=None):
        # 重置环境，返回初始状态
        # 初始化战斗机、防御弹和目标参数（速度、位置、制导律等）
        if not self.Isprint:
            self.fighter = copy.deepcopy(self.saves["figher"])
            self.defender = copy.deepcopy(self.saves["defender"])
            self.state = copy.deepcopy(self.saves["state"])
            self.t = self.t_0
            self.FD = relative(self.defender, self.fighter)
            self.FT = relative(self.fighter, self.target)

        self.success =False
        self.fail=False

        observation = self.state
        return observation , {}



    def step(self, action):
        # 根据动作更新状态，返回新状态、奖励、终止标志
        # 实现运动方程和防御弹逻辑
    
        # 按时间步长模拟飞行器运动

        

        # 战斗机
        a_y = (action[1]+1) * cos(action[0]) * 9.81
        a_z = (action[1]+1) * sin(action[0]) * 9.81 # 根据动作计算加速度

        self.fighter.update(self.dt, a_y , a_z)

        # 防御弹
        a_y , a_z = self.FD.proportional_navigation()
        self.defender.update(self.dt, a_y , a_z)  

        # 更新相对量
        self.FT.update()
        self.FD.update()
        

        r_1 , q_y1 , q_z1 = self.FD.state("defender")
        r_2 , q_y2 , q_z2 = self.FT.state("fighter")

        observation = np.array([r_1 , q_y1 , q_z1 , r_2 , q_y2 , q_z2])

        reward = self.calculate_reward()

        terminated = self.success or self.fail

        truncated = False

        if self.Isprint:
            self.update_plotdata()





        return observation, reward, terminated, truncated , {}





    def calculate_eta(self):
        v_f = self.fighter.velocity * np.array([cos(self.fighter.theta) * cos(self.fighter.psi),
                                           -cos(self.fighter.theta) * sin(self.fighter.psi),
                                            sin(self.fighter.theta)])  # 战机速度矢量
        l_ft = self.target.position - self.fighter.position
        self.eta_ft = acos((np.dot(v_f, l_ft) / (np.linalg.norm(v_f) * np.linalg.norm(l_ft)))) # 战机速度与目标视线夹角

    
    def calculate_reward(self):
        # 根据论文公式计算奖励（突防、被拦截、控制能量等）

        self.calculate_eta()




        
        if self.FT.r[1] <= R_FT and self.FD.dr > 0 and self.FD.r[1] > R_DAM and self.eta_ft <= ETA_FT:
            self.success = True
            return K_PLUS
        
        elif self.FD.r[1] <= R_FD and self.FD.dr > 0 and self.FD.r[1] > R_DAM and self.eta_ft > ETA_FT:
            return 0
        
        elif self.FD.r[1] < R_DAM:
            self.fail = True
            return K_MINUS
        
        else:
            if self.eta_ft <= ETA_FT:
                reward_1 = K_R1 * cos(self.eta_ft)
            else:
                reward_1 = -0.05
            
            if self.FD.dr < 0:
                reward_2 = -K_R2 * np.exp(-self.FD.r/100)
            else:
                reward_2 = 0
            
            reward_3 = -K_R3 * (self.fighter.a_y ** 2 + self.fighter.a_z ** 2)


            return reward_1 + reward_2 + reward_3




            

