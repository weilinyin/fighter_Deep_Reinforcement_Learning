from stable_baselines3.ppo import PPO
import torch.nn as nn
import torch as th
from math import cos , exp
from MyEnvs import relative
import numpy as np





class expert_generator:
    def __init__(self , FD:relative , FT:relative , t_0):

        self.t_0 = t_0
        self.FD = FD
        self.FT = FT
        self.c_0 = self.FD.dq_z


    def generate(self ,t , c_f):
        t_go =  - self.FD.r / self.FD.dr
        t_f = t + t_go - 0.7
        
        if t < t_f:

            k_FD = self.FD.dr / self.FD.r
            r_FD = self.FD.r
            q_FD = self.FD.q_z
            psi_VF = self.FD.target.psi

            
            
            

            a_E1 = 2 * k_FD * r_FD * exp(k_FD *(t_f - t)) / (1-exp(2*k_FD *(t_f - self.t_0)))
            a_E2 = (self.c_0 * exp(k_FD * (t_f - self.t_0)) - c_f) / cos(q_FD - psi_VF)

            a_E = a_E1 * a_E2 / (9.81*2)

            return a_E
        else:
            
            self.FT.proportional_navigation()
            _ , a_E = self.FT.calculate_a()

            return a_E /(9.81*2)




        

        
        


class GAILDiscriminator(nn.Module):
    """状态-动作对判别器"""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0] + action_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, states, actions):
        return self.net(th.cat([states, actions], dim=1))

class GAIL_PPO(PPO):
    def __init__(self, expert_generator, **kwargs):
        super().__init__(**kwargs)
        # 初始化判别器
        self.discriminator = GAILDiscriminator(
            self.observation_space, 
            self.action_space
        )
        self.disc_optimizer = th.optim.Adam(
            self.discriminator.parameters(), 
            lr=1e-3
        )
        self.expert_generator = expert_generator
        
    def _calc_gail_reward(self, obs, actions):
        """计算对抗奖励"""
        with th.no_grad():
            expert_prob = self.discriminator(obs, actions)
            return -th.log(1 - expert_prob + 1e-8)
    
    def train(self) -> None:
        # 收集智能体数据
        agent_obs, agent_acts, _ = self.collect_rollouts()
        
        # 获取专家数据（实时生成）
        expert_obs, expert_acts = self.expert_generator.sample()
        
        # 训练判别器
        for _ in range(5):  # 对抗训练次数
            # 计算判别损失
            agent_probs = self.discriminator(agent_obs, agent_acts)
            expert_probs = self.discriminator(expert_obs, expert_acts)
            loss = -th.log(expert_probs).mean() - th.log(1 - agent_probs).mean()
            
            # 更新判别器
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
        
        # 用对抗奖励替代原始奖励
        self.rollout_buffer.rewards = self._calc_gail_reward(
            th.as_tensor(self.rollout_buffer.observations),
            th.as_tensor(self.rollout_buffer.actions)
        )
        
        # 执行PPO原始训练流程[2,5](@ref)
        super().train()