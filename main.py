import gymnasium as gym
from gymnasium import spaces
import numpy as np

class aircraft:
    def __init__(self,position,theta,psi,velocity,a_max):
        self.position = position # 战机位置 (x, y, z)
        self.theta = theta # 弹道倾角 (theta)
        self.psi = psi # 弹道偏角 (psi)
        self.velocity = velocity # 速度 (v)
        self.a_max=a_max # 最大过载 (a_max)
    
    def simulate_motion(self, action):
        



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
