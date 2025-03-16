from MyEnvs import FighterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from math import log
import numpy as np


import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple

from custom_things import SmartStopCallback


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
            nn.Linear(feature_dim, 300), nn.ReLU() ,nn.Linear(300,300) , nn.Tanh() ,nn.Linear(300,last_layer_dim_pi) ,nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 300), nn.ReLU() , nn.Linear(300,300) , nn.ReLU() ,nn.Linear(300,last_layer_dim_vf) 
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

        self.log_std_init = log(0.5/(9.81*2))  # 固定log_std的初始值
        self.log_std = nn.Parameter(
            th.ones(2) * self.log_std_init, 
            requires_grad=False
        )
        
        
        # 禁用自动构建的mlp_extractor，替换为自定义网络
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim,2,300)





myenv = FighterEnv(False , 0.01 , 0.01)


myenv = Monitor(myenv)

callback = SmartStopCallback(target_reward=350 , avg_window=30 , stop_threshold=200)


model_1 = PPO(policy = CustomPolicy, env = myenv, verbose=1, device='cpu',learning_rate = 0.005,
              gae_lambda= 0.98 , gamma = 0.96 , n_steps = 2048 , batch_size = 256 , n_epochs = 4 ,clip_range = 0.2  ,normalize_advantage= False,  )

model_1.learn(total_timesteps=1e5, log_interval=1 ,callback = callback  , progress_bar=True)

model_1.save("model_1")
del model_1

np.save("fig/PPO三维仿真/Training_Progress.npy" , np.array(callback.episode_rewards))


# 绘制训练曲线
plt.figure(figsize=(12, 6))

# 绘制原始奖励（透明显示趋势）
plt.plot(callback.episode_rewards, alpha=0.2, label='Raw Reward')

# 绘制滑动平均奖励
if len(callback.mean_rewards) > 0:
    x_axis = np.arange(len(callback.mean_rewards)) + callback.avg_window
    plt.plot(x_axis, callback.mean_rewards, color='red', label='30-Episode Average')

# 标注停止训练的位置
if callback.consecutive_count >= callback.stop_threshold:
    stop_episode = len(callback.episode_rewards) - callback.stop_threshold
    plt.axvline(x=stop_episode, color='green', linestyle='--', 
               label='Training Stopped')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Training Progress (Stopped at Episode {len(callback.episode_rewards)})')
plt.legend()
plt.grid(True)
plt.savefig('fig/PPO三维仿真/Training_Progress.png')
plt.show()




