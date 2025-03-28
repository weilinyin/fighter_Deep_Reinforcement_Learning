import numpy as np
from numpy import sin , cos, tan , arccos ,arcsin ,arctan ,sqrt ,tanh
import gymnasium as gym
import copy



R_DAM = 20
R_FD = 18000
R_FT = 30000
R_CHANGE = 10000
ETA_FT = 10 * np.pi / 180
K_PLUS = 50
K_MINUS = -50
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

    def simulate(self , dt ,dtheta , dpsi):

        self.theta += dtheta * dt
        self.psi += dpsi * dt
        self.position[0] += self.velocity * cos(self.psi) * cos(self.theta) * dt
        self.position[1] += self.velocity * sin(self.theta) * dt
        self.position[2] += -self.velocity * sin(self.psi) * cos(self.theta) * dt



    



class relative:
    def __init__(self,chaser:aircraft,target:aircraft):
        # 初始化相对量

        self.chaser = chaser
        self.target = target

        self.calculate_relative_state()


        self.r_0 = self.r
        
        
        self.dq_y = 0.0
        self.dq_z = 0.0
        self.dr = 0.0
        self.dtheta = 0.0
        self.dpsi = 0.0

    def check_detection(self):
        # 检测战机是否探测到导弹
        if self.r < self.target.R:
            return True
        else:
            return False
        

    def calculate_relative_state(self):
        # 计算相对状态量
        r = self.target.position - self.chaser.position
        self.r = np.linalg.norm(r)
        self.q_y = arcsin(r[1] / self.r)
        self.q_z = arccos(r[0] / (np.linalg.norm(np.array([r[0] , r[2]]))))
        if r[2] > 0:
            self.q_z = -self.q_z
         





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
            eta_y = arcsin(eta_y)
            eta_z = arctan((cos(q_y) * sin(q_z) * cos(self.chaser.psi) -cos(q_y) * cos(q_z) * sin(self.chaser.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.chaser.theta) * cos(self.chaser.psi) +
                           sin(self.chaser.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.chaser.theta) * sin(self.chaser.psi)))
        elif type == "fighter":
            r_bar = (r-self.chaser.R) / (self.r_0 - self.chaser.R) 

            eta_y = (-sin(self.chaser.theta) * cos(self.chaser.psi) * cos(q_y) * cos(q_z) +
                          cos(self.chaser.theta) * sin(q_y) -
                          sin(self.chaser.theta) * cos(self.chaser.psi) * cos(q_y) * sin(q_z))
            eta_y = np.clip(eta_y, -1, 1)
            eta_z = arctan((cos(q_y) * sin(q_z) * cos(self.chaser.psi) -cos(q_y) * cos(q_z) * sin(self.chaser.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.chaser.theta) * cos(self.chaser.psi) +
                           sin(self.chaser.theta) * sin(q_y) +
                           cos(q_y) * sin(q_z) * cos(self.chaser.theta) * sin(self.chaser.psi)))
        eta_y = tanh(eta_y)
        eta_z = tanh(eta_z)
        return r_bar , eta_y , eta_z

    def state_2(self , type):
        if type == "defender":
            r_bar = self.r / self.chaser.R
        elif type == "fighter":
            r_bar = (self.r - self.chaser.R) / (self.r_0 - self.chaser.R)

        eta_z = tanh(self.q_z - self.chaser.psi)

        return r_bar , eta_z





    
    def proportional_navigation(self):
        # 比例导引法

        self.dtheta = 3 * self.dq_y * cos(self.q_z - self.chaser.psi)
        self.dpsi = 3 * self.dq_z - 3 * self.dq_y * tan(self.chaser.theta) * sin(self.q_z - self.chaser.psi)


    
    def calculate_a(self):

        self.a_y = self.dtheta * self.chaser.velocity
        self.a_z = -self.dpsi * self.chaser.velocity * cos(self.chaser.theta)

        a = sqrt(self.a_y**2 + self.a_z**2)


        if a >self.chaser.a_max:
            self.a_y = self.a_y * self.chaser.a_max / a
            self.a_z = self.a_z * self.chaser.a_max / a
            self.dtheta = self.a_y / self.chaser.velocity
            self.dpsi = -self.a_z / (self.chaser.velocity * cos(self.chaser.theta))

        return self.a_y , self.a_z







    def simulate(self,dt):
        # 模拟飞行器运动




        self.dr = (self.target.velocity * (cos(self.target.theta) * cos(self.q_y) * cos(self.target.psi - self.q_z) +
                                            sin(self.target.theta) * sin(self.q_y)) -
                    self.chaser.velocity * (cos(self.chaser.theta) * cos(self.q_y) * cos(self.chaser.psi - self.q_z) +
                                            sin(self.chaser.theta) * sin(self.q_y)))
            
        self.dq_y = (self.target.velocity * (sin(self.target.theta) * cos(self.q_y) -
                                            cos(self.target.theta) * sin(self.q_y) * cos(self.target.psi - self.q_z)) -
                        self.chaser.velocity * (sin(self.chaser.theta) * cos(self.q_y) -
                                            cos(self.chaser.theta) * sin(self.q_y) * cos(self.chaser.psi - self.q_z)))/self.r
        
        self.dq_z = (self.target.velocity * cos(self.target.theta) * sin(self.target.psi - self.q_z) -
                        self.chaser.velocity * cos(self.chaser.theta) * sin(self.chaser.psi - self.q_z))/(self.r * cos(self.q_y))
    

        _ , _ = self.calculate_a()


        # 更新位置
        
        
        self.chaser.simulate(dt , self.dtheta , self.dpsi)

        self.calculate_relative_state()





        













    


# 1. 环境模块
class FighterEnv(gym.Env):
    k_plus = 100
    k_minus = -100

    def __init__(self,Isprint = False , dt = 0.1 ,Dt = 1):
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
        self.dt_0 = dt
        self.Dt = Dt
        self.dt = Dt

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
        self.saves = {"figher":copy.deepcopy(self.fighter),"defender":copy.deepcopy(self.defender)
                      ,"state":copy.deepcopy(self.state),"FD":copy.deepcopy(self.FD),"FT":copy.deepcopy(self.FT)}

        # 初始化成功突防标志
        self.success = False



        # 初始化失败标志
        self.fail = False

        # 初始化总回合数
        self.episode = 0





    def start_simulate(self):

        while not self.FD.check_detection():

            self.FT.proportional_navigation()
            self.FD.proportional_navigation()           
            
            self.FD.simulate(self.dt)
            self.FT.simulate(self.dt) 
            

            self.t += self.dt

            
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
        self.plotdata["fighter"]["a_y"].append(self.FT.a_y)
        self.plotdata["fighter"]["a_z"].append(self.FT.a_z)
        self.plotdata["fighter"]["r"].append(self.FT.r)

        self.plotdata["defender"]["x"].append(self.defender.position[0])
        self.plotdata["defender"]["y"].append(self.defender.position[1])
        self.plotdata["defender"]["z"].append(self.defender.position[2])
        self.plotdata["defender"]["theta"].append(self.defender.theta)
        self.plotdata["defender"]["psi"].append(self.defender.psi)
        self.plotdata["defender"]["a_y"].append(self.FD.a_y)
        self.plotdata["defender"]["a_z"].append(self.FD.a_z)
        self.plotdata["defender"]["r"].append(self.FD.r)

        self.plotdata["eta"].append(self.eta_ft)

        self.t_array.append(self.t)
        




    def reset(self , seed=None, options=None):
        # 重置环境，返回初始状态
        # 初始化战斗机、防御弹和目标参数（速度、位置、制导律等）
        
        self.fighter = copy.deepcopy(self.saves["figher"])
        self.defender = copy.deepcopy(self.saves["defender"])
        self.state = copy.deepcopy(self.saves["state"])
        
            
        self.FD = relative(self.defender, self.fighter)
        self.FT = relative(self.fighter, self.target)

        self.FD.dr = self.saves["FD"].dr
        self.FD.dq_y = self.saves["FD"].dq_y
        self.FD.dq_z = self.saves["FD"].dq_z
        self.FD.dtheta = self.saves["FD"].dtheta
        self.FD.dpsi = self.saves["FD"].dpsi

        self.FT.dr = self.saves["FT"].dr
        self.FT.dq_y = self.saves["FT"].dq_y
        self.FT.dq_z = self.saves["FT"].dq_z
        self.FT.dtheta = self.saves["FT"].dtheta
        self.FT.dpsi = self.saves["FT"].dpsi


        self.t = self.t_0
        self.episode += 1

        self.success =False
        self.fail=False

        observation = self.state
        return observation , {}



    def step(self, action):
        # 根据动作更新状态，返回新状态、奖励、终止标志
        # 实现运动方程和防御弹逻辑
    
        # 按时间步长模拟飞行器运动

        # 防御弹
        if self.FD.r <= R_CHANGE:
            self.dt = self.dt_0
        else:
            self.dt = self.Dt

        self.t += self.dt
        
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

        
        self.FD.proportional_navigation()


        self.FT.simulate(self.dt)
        self.FD.simulate(self.dt)
        

        

        

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
        l_ft = self.target.position - self.fighter.position  # 目标与战斗机的相对位置矢量


        if np.linalg.norm(l_ft) == 0:
            self.eta_ft = 0
        else:
            self.eta_ft = arccos(np.clip(np.dot(v_f, l_ft) / (np.linalg.norm(v_f) * np.linalg.norm(l_ft)) , -1.0, 1.0)) # 战机速度与目标视线夹角

    
    def calculate_reward(self):
        # 根据论文公式计算奖励（突防、被拦截、控制能量等）

        self.calculate_eta()




        
        if self.FT.r <= R_FT and self.FD.dr > 0 and self.FD.r > R_DAM and self.eta_ft <= ETA_FT:
            self.success = True
            return self.k_plus
        
        elif self.FD.r <= R_FD and self.FD.dr > 0 and self.FD.r > R_DAM and self.eta_ft > ETA_FT:
            return 0
        
        elif self.FD.r < R_DAM or self.t > 2e3:
            self.fail = True
            return self.k_minus
        
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


class FighterEnv_2D(FighterEnv):
    k_plus = 50
    k_minus = -50
    def __init__(self, Isprint = False ,dt = 0.1 ,Dt = 1):
        super(FighterEnv, self).__init__()

        self.Isprint = Isprint

        
        self.observation_space = gym.spaces.Box(low = np.array([0  , -1 , -np.inf , -1]) ,
                                                high = np.array([1.1  , 1 , 2  , 1]) ,
                                                shape= (4,),
                                                dtype = np.float64)
        
        self.action_space = gym.spaces.Box(low = -1 ,
                                           high = 1 ,
                                           shape = (1,),
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
            np.array([40000,10000, -5000]).astype(np.float64),
            0,
            3.23,
            1000,
            5*9.81,
            R_FD
        )
        self.target = aircraft(
            np.array([50000,10000,0]).astype(np.float64),
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
        self.dt_0 = dt
        self.Dt = Dt
        self.dt = Dt

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
        r_1  , q_z1 = self.FD.state_2("defender")
        r_2  , q_z2 = self.FT.state_2("fighter")


        self.state = np.array([r_1  , q_z1 , r_2  , q_z2])
        # 存档
        self.saves = {"figher":copy.deepcopy(self.fighter),"defender":copy.deepcopy(self.defender)
                      ,"state":copy.deepcopy(self.state),"FD":copy.deepcopy(self.FD),"FT":copy.deepcopy(self.FT)}

        # 初始化成功突防标志
        self.success = False



        # 初始化失败标志
        self.fail = False

        # 初始化总回合数
        self.episode = 0

    def step(self,action):


        if self.FD.r <= R_CHANGE:
            self.dt = self.dt_0
        else:
            self.dt = self.Dt

        self.t += self.dt


        self.a_y = 0
        a_z = action[0] * 9.81 * 2

        self.a_z = a_z



        self.FT.dtheta = self.a_y/self.fighter.velocity
        self.FT.dpsi = -self.a_z/(self.fighter.velocity * cos(self.fighter.theta))

        
        
        self.FD.proportional_navigation()

        self.FT.simulate(self.dt)  
        self.FD.simulate(self.dt)
        

        

        

        r_1  , q_z1 = self.FD.state_2("defender")
        r_2  , q_z2 = self.FT.state_2("fighter")

        observation = np.array([r_1  , q_z1 , r_2  , q_z2])

        reward = self.calculate_reward()

        terminated = self.success or self.fail

        truncated = False

        if self.Isprint:
            self.update_plotdata()

        


        



        return observation, reward, terminated, truncated , {}

            

class FighterEnv_nopolicy(FighterEnv):

    def start_simulate(self):
        while self.FD.r > R_DAM and self.FT.r > 5000 and self.t < 30: # 不进行突防仿真

            self.FT.proportional_navigation()
            self.FD.proportional_navigation()          
            
            self.FD.simulate(self.dt)
            self.FT.simulate(self.dt)
             
            
            
            self.t += self.dt

            if self.Isprint:
                
                # 加入观察数据
                self.calculate_eta()
                self.update_plotdata()

class FighterEnv_nopolicy_2D(FighterEnv_2D):

    def start_simulate(self):

        while self.FD.r > R_DAM and self.FT.r > 5000 and self.t < 30: # 不进行突防仿真
            

                     
            

            
            self.FT.proportional_navigation()
            self.FD.proportional_navigation() 

            self.FD.simulate(self.dt)
            self.FT.simulate(self.dt)
             
            
            
            self.t += self.dt

            if self.Isprint:
                
                # 加入观察数据
                self.calculate_eta()
                self.update_plotdata()