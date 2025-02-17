import torch
import torch.nn as nn
import numpy as np
from math import sin , cos
from collections import deque

R_DAM = 20
R_FD = 18000
R_FT = 30000
ETA_FT = 10 * np.pi / 180


class aircraft:
    def __init__(self,position,theta,psi,velocity,a_max):
        self.position = position # 位置 (x, y, z)
        self.theta = theta # 弹道倾角 (theta)
        self.psi = psi # 弹道偏角 (psi)
        self.velocity = velocity # 速度 (v)
        self.a_max=a_max # 最大过载 (a_max)

class relative:
    def __init__(self,chaser:aircraft,target:aircraft):
        #初始化相对量
        self.r, self.q_y, self.q_z =self.state(self,chaser,target)
        self.chaser = chaser
        self.target = target
        self.theta = self.chaser.theta
        self.psi = self.chaser.psi

    def state(self,chaser:aircraft,target:aircraft):
        r = np.abs(target.position - chaser.position)
        q_y = np.asin((target.position[1] - chaser.position[1]) / r)
        q_z = np.acos((target.position[0] - chaser.position[0]) / (np.abs(target.position[0:1] - chaser.position[0:1])))
        r_bar = r / R_FD
        eta_y = np.arcsin(-sin(self.theta) * cos(self.psi) * cos(q_y) * cos(q_z) +
                          cos(self.theta) * sin(q_y) -
                          sin(self.theta) * cos(self.psi) * cos(q_y) * sin(q_z))
        eta_z = np.arctan((cos(q_y) * sin(q_z) * cos(self.psi) -cos(q_y) * cos(q_z) * sin(self.psi))/
                          (cos(q_y) * cos(q_z) * cos(self.theta) * ))

        return 


    
    def simulate(self,dt,a_y,a_z,action:bool):
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
        if action == False:
            #比例导引法
            self.dtheta = 3 * self.dq_y * cos(self.q_z - self.chaser.psi)
            self.dpsi = 3 * self.dq_z - 3 * self.dq_y * tan(self.chaser.theta) * sin(self.q_z - self.chaser.psi)
        else:
            self.dtheta = a_y/self.chaser.velocity
            self.dpsi = -a_z/(self.chaser.velocity * cos(self.chaser.theta))

        self.r += self.dr * dt
        self.q_y += self.dq_y * dt
        self.q_z += self.dq_z * dt
        self.chaser.psi += self.dpsi * dt
        self.chaser.theta += self.dtheta * dt

        return self.r , self.q_y , self.q_z
    


# 1. 环境模块
class FighterEnv:
    def __init__(self):
        # 初始化战斗机、防御弹和目标参数（速度、位置、制导律等）
        self.fighter = aircraft(
            np.array([0, 10000, 0]),
            0,
            0,
            900,
            2
        )
        self.defender = aircraft(
            np.array([40000, 10000, -5000]),
            0,
            3.23,
            1000,
            5
        )
        self.target = aircraft(
            np.array([50000,10000,0]),
            0,
            0,
            0,
            0
        )


        #初始化相对量
        self.FD = relative(self.defender, self.target)
        self.FT = relative(self.fighter, self.target)

        # 初始化状态
        r_1 , q_y1 , q_z1 = self.FD.state()
        r_2 , q_y2 , q_z2 = self.FT.state()

        self.state = np.array([
        ])




    
    def reset(self):
        # 重置环境，返回初始状态
        self.FD = relative(self.defender, self.target)
        self.FT = relative(self.fighter, self.target)
        return np.zeros(self.state_dim)
    
    def step(self, action):
        # 根据动作更新状态，返回新状态、奖励、终止标志
        # 实现运动方程和防御弹逻辑
        next_state = ...
        reward = self.calculate_reward()
        done = self.check_done()
        return next_state, reward, done
    
    def calculate_reward(self):
        # 根据论文公式计算奖励（突防、被拦截、控制能量等）
        return reward

# 2. 神经网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.fc(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        return self.fc(state)

# 3. PPO算法核心
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': 0.002},
            {'params': self.critic.parameters(), 'lr': 0.002}
        ])
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_mean = self.actor(state)
        dist = torch.distributions.Normal(action_mean, 0.5)  # 固定标准差
        action = dist.sample()
        return action.detach().numpy()
    
    def update(self, buffer):
        # 从buffer中采样，计算GAE和损失函数
        # 实现PPO的Clip损失和Critic的MSE损失
        pass

# 4. 主训练流程
env = FighterEnv()
ppo = PPO(state_dim=6, action_dim=2)
buffer = deque(maxlen=2048)

for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = ppo.select_action(state)
        next_state, reward, done = env.step(action)
        buffer.append((state, action, reward, next_state, done))
        state = next_state
    
    # 每收集完一个episode，更新网络
    ppo.update(buffer)

# 5. GAIL-PPO扩展（部分代码）
class Discriminator(nn.Module):
    def __init__(self, state_action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state_action):
        return self.fc(state_action)

class GAIL_PPO(PPO):
    def __init__(self, state_dim, action_dim, expert_data):
        super().__init__(state_dim, action_dim)
        self.discriminator = Discriminator(state_dim + action_dim)
        self.expert_data = expert_data  # 专家数据（状态-动作对）
    
    def update_discriminator(self, agent_data):
        # 训练判别器区分专家数据和智能体数据
        expert_loss = -torch.log(self.discriminator(self.expert_data))
        agent_loss = -torch.log(1 - self.discriminator(agent_data))
        total_loss = expert_loss + agent_loss
        total_loss.backward()

# 6. 迁移学习（TRL）
class TRL(PPO):
    def __init__(self, source_actor_2d, state_dim_3d, action_dim_3d):
        super().__init__(state_dim_3d, action_dim_3d)
        self.source_actor_2d = source_actor_2d  # 预训练的二维Actor
    
    def map_action(self, state_3d):
        # 将三维状态映射到二维，调用源Actor生成动作，再扩展为三维动作
        state_2d = state_3d[[0, 1, 3, 4]]  # 示例映射
        action_2d = self.source_actor_2d(state_2d)
        action_3d = np.zeros(2)
        action_3d[1] = action_2d  # 侧向加速度
        return action_3d