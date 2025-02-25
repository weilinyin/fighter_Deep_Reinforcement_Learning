from MyEnvs import FighterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from math import log

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

        self.log_std_init = 0  # 固定log_std的初始值
        #self.log_std = nn.Parameter(
            #th.ones(2) * self.log_std_init, 
            #requires_grad=False
        #)
        
        
        # 禁用自动构建的mlp_extractor，替换为自定义网络
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim,2,300)



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





myenv = FighterEnv()


myenv = Monitor(myenv)

callback = EpisodeRewardCallback()

model_1 = PPO(policy = CustomPolicy, env = myenv, verbose=1, device='cpu',learning_rate = 0.005,
              gae_lambda= 0.98 , gamma = 0.96 , n_steps = 2048 , batch_size = 256 , n_epochs = 4 ,clip_range = 0.2  )

model_1.learn(total_timesteps=1e6, log_interval=4 ,callback = callback )

model_1.save("model_1")
del model_1

# 绘图
plt.plot(callback.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Episode Rewards')
plt.savefig('fig\Reward_per_episode.png')
plt.show()




