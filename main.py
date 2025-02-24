from MyEnvs import FighterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium import ActionWrapper , spaces
import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union



class CustomNetwork(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    



class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # 禁用自动构建的mlp_extractor，替换为自定义网络

    def _build_mlp_extractor(self) -> None:
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


policy_kwargs = dict(activation_fn=nn.Tanh,
                     net_arch=dict(pi=[300, 300], vf=[300, 300]))


myenv = FighterEnv()

myenv = GaussianNoiseWrapper(myenv, noise_std=0.5)

myenv = Monitor(myenv)

callback = EpisodeRewardCallback()

model_1 = PPO("MlpPolicy", env = myenv, verbose=1, device='cpu',learning_rate = 0.005,
              gae_lambda= 0.98 , gamma = 0.96 , n_steps = 2048 , batch_size = 256 , n_epochs = 4 ,clip_range = 0.2 , policy_kwargs = policy_kwargs  )

model_1.learn(total_timesteps=4e5, log_interval=4 ,callback = callback)

model_1.save("model_1")
del model_1

# 绘图
plt.plot(callback.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Episode Rewards')
plt.savefig('fig\Reward_per_episode.png')
plt.show()




