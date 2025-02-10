import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import sin , cos

def rotation(axis,angle):
    # 旋转矩阵计算
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return R


class aircraft:
    def __init__(self,position,theta,psi,velocity,a_max):
        self.position = position # 位置 (x, y, z)
        self.theta = theta # 弹道倾角 (theta)
        self.psi = psi # 弹道偏角 (psi)
        self.velocity = velocity # 速度 (v)
        self.a_max=a_max # 最大过载 (a_max)
    
class relative:
    def __init__(self,chaser:aircraft,target:aircraft):
        self.r = np.abs(target.position - chaser.position)
        self.q_y = m.asin((target.position[1] - chaser.position[1]) / self.r)
        self.q_z = m.acos((target.position[0] - chaser.position[0]) / (np.abs(target.position[0:1] - chaser.position[0:1])))
        self.chaser = chaser
        self.target = target

    
    def simulate(self,dt,action):
        self.dr = (self.target.velocity * (m.cos(self.target.theta) * m.cos(self.q_y) * m.cos(self.target.psi - self.q_z) +
                                            m.sin(self.target.theta) * m.sin(self.q_y)) -
                    self.chaser.velocity * (m.cos(self.chaser.theta) * m.cos(self.q_y) * m.cos(self.chaser.psi - self.q_z) +
                                            m.sin(self.chaser.theta) * m.sin(self.q_y)))
        self.dq_y = (self.target.velocity * (m.sin(self.target.theta) * m.cos(self.q_y) -
                                             m.cos(self.target.theta) * m.sin(self.q_y) * m.cos(self.target.psi - self.q_z) +
                                             m.sin(self.target.psi - self.q_z) * m.sin(self.q_y)) -
                     self.chaser.velocity * (m.sin(self.chaser.theta) * m.cos(self.q_y) -
                                             m.cos(self.chaser.theta) * m.sin(self.q_y) * m.cos(self.chaser.psi - self.q_z) +
                                             m.sin(self.chaser.psi - self.q_z) * m.sin(self.q_y)))/self.r
        self.dq_z = (self.target.velocity * m.cos(self.target.theta) * m.sin(self.chaser.psi - self.q_z) -
                     self.chaser.velocity * m.cos(self.chaser.theta) * m.sin(self.chaser.psi - self.q_z))/(self.r * m.cos(self.q_y))
        if action == None
            self.dtheta = 3 * self.dq_y * 










    


class CustomEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(
            low = -2,
            high = 2,
            shape = (1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low = np.array([1, -1, 0, 0, -1]),
            high = np.array([np.inf, 1, 1, 1, 1]),
            dtype = np.float32
        )
        self.state=None
    def reset(self, seed=None, options=None):
        # 重置环境状态，返回初始观察
        fighter = aircraft(np.array([0, 10000, 0]), 0, 0, 900, 2)
        defender = aircraft(np.array([40000, 10000, -5]), 0, 3.23, 1000, 2)
        target = aircraft(np.array([50000, 10000, 0]), 0, 0, 1000, 2)
        return observation, {}

    def step(self, action):
        # 执行动作，返回 (observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info
