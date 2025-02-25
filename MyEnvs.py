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
DT = 1
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
    



class relative:
    def __init__(self,chaser:aircraft,target:aircraft):
        # 初始化相对量

        self.r = np.linalg.norm(target.position - chaser.position)
        self.r_0 = self.r
        self.q_y = asin((target.position[1] - chaser.position[1]) / self.r)
        self.q_z = acos((target.position[0] - chaser.position[0]) / (np.linalg.norm(np.array([target.position[0] - chaser.position[0] , target.position[2] - chaser.position[2]])))) 
        self.dq_y = 0
        self.dq_z = 0


        self.dr = 0
        self.dq_y = 0
        self.dq_z = 0

        self.chaser = chaser
        self.target = target
        self.theta = self.chaser.theta
        self.psi = self.chaser.psi
        self.dtheta = 0
        self.dpsi = 0

    def check_detection(self):
        # 检测战机是否探测到导弹
        if self.r < self.target.R:
            return True
        else:
            return False



    def state(self,type = "fighter"):
        # 输出相对状态量

        r = self.r
        q_y = self.q_y
        q_z = self.q_z

       # 归一化处理
        if type == "defender":
            r_bar = r / self.chaser.R
            eta_y = (-sin(self.target.theta) * cos(self.target.psi) * cos(q_y) * cos(q_z) +
                          cos(self.target.theta) * sin(q_y) -
                          sin(self.target.theta) * cos(self.target.psi) * cos(q_y) * sin(q_z))
            eta_y = np.clip(eta_y, -1, 1)
            eta_y = asin(eta_y)
            eta_z = atan((cos(q_y) * sin(q_z) * cos(self.psi) -cos(q_y) * cos(q_z) * sin(self.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.theta) * cos(self.psi) +
                           sin(self.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.theta) * sin(self.psi)))
        elif type == "fighter":
            r_bar = (r-self.chaser.R) / (self.r_0 - self.chaser.R) 

            eta_y = (-sin(self.theta) * cos(self.psi) * cos(q_y) * cos(q_z) +
                          cos(self.theta) * sin(q_y) -
                          sin(self.theta) * cos(self.psi) * cos(q_y) * sin(q_z))
            eta_y = np.clip(eta_y, -1, 1)
            eta_z = atan((cos(q_y) * sin(q_z) * cos(self.psi) -cos(q_y) * cos(q_z) * sin(self.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.theta) * cos(self.psi) +
                           sin(self.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.theta) * sin(self.psi)))
        eta_y = tanh(eta_y)
        eta_z = tanh(eta_z)
        return r_bar , eta_y , eta_z




    
    def proportional_navigation(self):
        # 比例导引法

        self.dtheta = 3 * self.dq_y * cos(self.q_z - self.psi)
        self.dpsi = 3 * self.dq_z - 3 * self.dq_y * tan(self.theta) * sin(self.q_z - self.psi)


    
    def calculate_a(self):

        self.a_y = self.dtheta * self.chaser.velocity
        self.a_z = -self.dpsi * self.chaser.velocity * cos(self.theta)

        a = sqrt(self.a_y**2 + self.a_z**2)


        if a >self.chaser.a_max:
            self.a_y = self.a_y * self.chaser.a_max / a
            self.a_z = self.a_z * self.chaser.a_max / a
            self.dtheta = self.a_y / self.chaser.velocity
            self.dpsi = -self.a_z / (self.chaser.velocity * cos(self.theta))






            











    def simulate(self,dt):
        # 模拟飞行器运动




        self.dr = (self.target.velocity * (cos(self.target.theta) * cos(self.q_y) * cos(self.target.psi - self.q_z) +
                                            sin(self.target.theta) * sin(self.q_y)) -
                    self.chaser.velocity * (cos(self.theta) * cos(self.q_y) * cos(self.psi - self.q_z) +
                                            sin(self.theta) * sin(self.q_y)))
            
        self.dq_y = (self.target.velocity * (sin(self.target.theta) * cos(self.q_y) -
                                            cos(self.target.theta) * sin(self.q_y) * cos(self.target.psi - self.q_z)) -
                        self.chaser.velocity * (sin(self.theta) * cos(self.q_y) -
                                            cos(self.theta) * sin(self.q_y) * cos(self.psi - self.q_z)))/self.r
        
        self.dq_z = (self.target.velocity * cos(self.target.theta) * sin(self.psi - self.q_z) -
                        self.chaser.velocity * cos(self.theta) * sin(self.psi - self.q_z))/(self.r * cos(self.q_y))
        

        self.r += self.dr * dt
        self.q_y += self.dq_y * dt
        self.q_z += self.dq_z * dt

        self.calculate_a()

        self.theta += self.dtheta * dt 
        self.psi += self.dpsi * dt

        # 更新位置
        
        self.chaser.position[0] = self.target.position[0] - self.r * cos(self.q_y) * cos(self.q_z)
        self.chaser.position[1] = self.target.position[1] - self.r * sin(self.q_y)
        self.chaser.position[2] = self.target.position[2] + self.r * cos(self.q_y) * sin(self.q_z)
        self.chaser.theta = self.theta
        self.chaser.psi = self.psi




        













    


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
                                           dtype = np.float64) 
        self.fighter = aircraft(
            np.array([0, 10000, 0]).astype(np.float64),
            0,
            0,
            900,
            2*9.81,
            R_FT
        )
        self.defender = aircraft(
            np.array([40000, 5000, -5000]).astype(np.float64),
            0.11,
            3.23,
            1000,
            5*9.81,
            R_FD
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
            
            
            self.FT.simulate(self.dt) 
            self.FD.simulate(self.dt)

            self.FT.proportional_navigation()
            self.FD.proportional_navigation()
            
            # 加入观察数据
            if self.Isprint:
                self.calculate_eta()
                self.update_plotdata()



    def update_plotdata(self):
        self.plotdata["fighter"]["x"].append(self.fighter.position[0])
        self.plotdata["fighter"]["y"].append(self.fighter.position[1])
        self.plotdata["fighter"]["z"].append(self.fighter.position[2])
        self.plotdata["fighter"]["theta"].append(self.FT.theta)
        self.plotdata["fighter"]["psi"].append(self.FT.psi)
        self.plotdata["fighter"]["a_y"].append(self.FT.a_y)
        self.plotdata["fighter"]["a_z"].append(self.FT.a_z)
        self.plotdata["fighter"]["r"].append(self.FT.r)

        self.plotdata["defender"]["x"].append(self.defender.position[0])
        self.plotdata["defender"]["y"].append(self.defender.position[1])
        self.plotdata["defender"]["z"].append(self.defender.position[2])
        self.plotdata["defender"]["theta"].append(self.FD.theta)
        self.plotdata["defender"]["psi"].append(self.FD.psi)
        self.plotdata["defender"]["a_y"].append(self.FD.a_y)
        self.plotdata["defender"]["a_z"].append(self.FD.a_z)
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

        # 防御弹
        if self.FD.r <= 5000:
            self.dt = 0.1
        else:
            self.dt = DT

        self.FD.simulate(self.dt)
        self.FD.proportional_navigation()

        a_y = action[0] * 9.81 * 2
        a_z = action[1] * 9.81 * 2

        a = sqrt(a_y**2 + a_z**2)


        if a > self.fighter.a_max:
            self.a_y = a_y * self.fighter.a_max / a
            self.a_z = a_z * self.fighter.a_max / a
        else:
            self.a_y = a_y
            self.a_z = a_z



        self.FT.dtheta = self.a_y/self.fighter.velocity
        self.FT.dpsi = -self.a_z/(self.fighter.velocity * cos(self.fighter.theta))

        self.FT.simulate(self.dt)  # 更新战斗机状态


        

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
        v_f = self.fighter.velocity * np.array([cos(self.FT.theta) * cos(self.FT.psi),
                                           -cos(self.FT.theta) * sin(self.FT.psi),
                                            sin(self.FT.theta)])  # 战机速度矢量
        l_ft = self.FT.r * np.array([cos(self.FT.q_y) * cos(self.FT.q_z),
                                    -cos(self.FT.q_y) * sin(self.FT.q_z),
                                    sin(self.FT.q_y)])
        self.eta_ft = acos((np.dot(v_f, l_ft) / (np.linalg.norm(v_f) * np.linalg.norm(l_ft)))) # 战机速度与目标视线夹角

    
    def calculate_reward(self):
        # 根据论文公式计算奖励（突防、被拦截、控制能量等）

        self.calculate_eta()




        
        if self.FT.r <= R_FT and self.FD.dr > 0 and self.FD.r > R_DAM and self.eta_ft <= ETA_FT:
            self.success = True
            return K_PLUS
        
        elif self.FD.r <= R_FD and self.FD.dr > 0 and self.FD.r > R_DAM and self.eta_ft > ETA_FT:
            return 0
        
        elif self.FD.r < R_DAM:
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
            
            reward_3 = -K_R3 * (self.a_y ** 2 + self.a_z ** 2)


            return reward_1 + reward_2 + reward_3




            

class FighterEnv_2(FighterEnv):

    def start_simulate(self):
        while self.FD.r > R_DAM and self.FT.r > 0: # 不进行突防仿真
            
            
            self.FT.simulate(self.dt) 
            self.FD.simulate(self.dt)

            self.FT.proportional_navigation()
            self.FD.proportional_navigation()

            if self.Isprint:
            
                # 加入观察数据
                self.calculate_eta()
                self.update_plotdata()