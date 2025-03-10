from stable_baselines3.ppo import PPO
import torch.nn as nn
import torch as th
from math import cos , exp
from MyEnvs import relative
import numpy as np
from MyEnvs import FighterEnv_2D
from stable_baselines3.common.buffers import RolloutBuffer
from typing import Optional
from collections.abc import Generator
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize



class CustomRolloutBuffer(RolloutBuffer):
    #魔改了buffer，增加了专家策略
    expert_actions:np.ndarray
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "expert_actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.expert_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        expert_action: np.ndarray,  # Added parameter for expert action
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.expert_actions[self.pos] = np.array(expert_action)  # Add expert action to the buffer
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.expert_actions[batch_inds],  # Include expert actions in the samples
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


















class expert_generator:
    def __init__(self ,env:FighterEnv_2D ,c_f=-0.002):
        self.env = env
        self.t_0 = env.t_0
        self.c_0 = self.env.FD.dq_z
        self.t_f = self.t_0 - self.env.FD.r / self.env.FD.dr -0.5
        self.c_f = c_f


    def generate(self ,t):
        
        if t < self.t_f:

            k_FD = self.env.FD.dr / self.env.FD.r
            r_FD = self.env.FD.r
            q_FD = self.env.FD.q_z
            psi_VF = self.env.FD.target.psi

            
            
            

            a_E1 = 2 * k_FD * r_FD * exp(k_FD *(self.t_f - t)) / (1-exp(2*k_FD *(self.t_f - self.t_0)))
            a_E2 = (self.c_0 * exp(k_FD * (self.t_f - self.t_0)) - self.c_f) / cos(q_FD - psi_VF)

            a_E = a_E1 * a_E2 / (9.81*2)

            return a_E
        else:
            
            self.env.FT.proportional_navigation()
            _ , a_E = self.env.FT.calculate_a()

            return a_E /(9.81*2)


    def sample(self):
        a = self.generate(self.env.t)

        r_1 , q_y1 , q_z1 = self.env.FD.state("defender")
        r_2 , q_y2 , q_z2 = self.env.FT.state("fighter")
        obs = np.array([r_1 , q_y1 , q_z1 , r_2 , q_y2 , q_z2])

        return obs , a
        

        

        
        


class GAILDiscriminator(nn.Module):
    """状态-动作对判别器"""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0] + action_space.shape[0], 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
    
    def forward(self, states, actions):
        return self.net(th.cat([states, actions], dim=1))

class GAIL_PPO(PPO):
    def __init__(self, expert_generator:expert_generator, **kwargs):
        super().__init__(rollout_buffer_class=CustomRolloutBuffer, **kwargs)
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
        expert_obs, expert_acts = self.expert_generator.sample(agent_obs)
        
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