from MyEnvs import FighterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium import ActionWrapper
import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 300),
            nn.ReLU()
        )
        # Actor分支
        self.actor = nn.Sequential(
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 2),
            nn.Tanh()
        )
        # Critic分支
        self.critic = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
    
    def forward(self, features):
        shared_out = self.shared(features)
        return self.actor(shared_out), self.critic(shared_out)
    
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # 禁用自动构建的mlp_extractor，替换为自定义网络
        self.mlp_extractor = CustomNetwork(self.features_dim)


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 遍历所有子环境的info
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
        return True

class GaussianNoiseWrapper(ActionWrapper):
    def __init__(self, env, noise_std=0.1):
        super(GaussianNoiseWrapper, self).__init__(env)
        self.noise_std = noise_std

    def action(self, action):
        # 添加高斯噪声
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return action


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[300, 300], vf=[300, 300]))


myenv = FighterEnv()

myenv = GaussianNoiseWrapper(myenv, noise_std=0.5)

myenv = Monitor(myenv)

callback = EpisodeRewardCallback()

model_1 = PPO("MlpPolicy", env = myenv, verbose=1, device='cpu',learning_rate = 0.005,
              gae_lambda= 0.98 , gamma = 0.96 , n_steps = 2048 , batch_size = 256 , n_epochs = 4 ,clip_range = 0.2 ,policy_kwargs=policy_kwargs)

model_1.learn(total_timesteps=4e5, log_interval=4 ,callback = callback)


# 绘图
plt.plot(callback.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Episode Rewards')
plt.show()

model_1.save("model_1")
del model_1
