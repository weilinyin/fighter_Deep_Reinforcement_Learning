from MyEnvs import FighterEnv_2D
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from math import log
import numpy as np


import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple



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
            nn.Linear(feature_dim, 160), nn.ReLU() ,nn.Linear(160,160) , nn.Tanh() ,nn.Linear(160,last_layer_dim_pi) ,nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU() , nn.Linear(200,200) , nn.ReLU() ,nn.Linear(200,last_layer_dim_vf) 
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

        self.log_std_init = log(0.5)  # 固定log_std的初始值
        self.log_std = nn.Parameter(
            th.ones(1) * self.log_std_init, 
            requires_grad=False
        )
        
        
        # 禁用自动构建的mlp_extractor，替换为自定义网络
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim,2,300)



class SmartStopCallback(BaseCallback):
    def __init__(self, 
                 avg_window=30,         # 平均奖励计算窗口
                 stop_threshold=200,     # 奖励达标连续次数
                 target_reward=300,     # 奖励阈值
                 verbose=0):
        super().__init__(verbose)
        self.avg_window = avg_window
        self.stop_threshold = stop_threshold
        self.target_reward = target_reward
        self.episode_rewards = []        # 原始奖励记录
        self.mean_rewards = []           # 滑动平均奖励
        self.consecutive_count = 0       # 连续达标计数器

    def _on_step(self) -> bool:
        # 收集所有环境的episode奖励
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                
                # 计算滑动平均
                if len(self.episode_rewards) >= self.avg_window:
                    avg_reward = np.mean(self.episode_rewards[-self.avg_window:])
                    self.mean_rewards.append(avg_reward)
                    
                    # 检查停止条件
                    if avg_reward >= self.target_reward:
                        self.consecutive_count += 1
                        if self.consecutive_count >= self.stop_threshold:
                            self.model.stop_training = True  # 触发停止训练
                            return False
                    else:
                        self.consecutive_count = 0  # 重置计数器
        return True





#myenv = DummyVecEnv([lambda: FighterEnv_2D() for _ in range(32)])

myenv = FighterEnv_2D(dt = 0.025 ,Dt = 0.025)
myenv = Monitor(myenv)

callback = SmartStopCallback(target_reward=60 , avg_window=30 , stop_threshold=200)


model = PPO(policy = CustomPolicy, env = myenv, verbose=1, device='cpu',learning_rate = 0.002,
              gae_lambda= 0.97 , gamma = 0.97 , n_steps = 2048 , batch_size = 512 , n_epochs = 4 ,clip_range = 0.2  )

model.learn(total_timesteps=1e7, log_interval=1 ,callback = callback , progress_bar= True )

model.save("model_2D")
del model

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
plt.savefig('fig\PPO二维仿真\Training_Progress.png')
plt.show()