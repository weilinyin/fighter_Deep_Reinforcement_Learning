import torch
import torch.nn as nn
import numpy as np
from math import sin , cos
from collections import deque
from MyEnvs import FighterEnv


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