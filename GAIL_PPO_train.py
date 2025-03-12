from MyEnvs import FighterEnv_2D
from GAIL_PPO import GAIL_PPO ,expert_generator
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt
from custom_things import SmartStopCallback , CustomPolicy





myenv = FighterEnv_2D(dt = 0.01 ,Dt = 0.01)
expert = expert_generator(env=myenv)
myenv = Monitor(myenv)

callback = SmartStopCallback(target_reward=200 , avg_window=30 , stop_threshold=200)




model = GAIL_PPO(policy = CustomPolicy, env = myenv, verbose=1, device='cpu',learning_rate = 0.002,
              gae_lambda= 0.97 , gamma = 0.97 , n_steps = 2048 , batch_size = 512 , n_epochs = 4 ,clip_range = 0.2   ,expert_generator=expert , N_gail=300)

model.learn(total_timesteps=1e7, log_interval=1 ,callback = callback , progress_bar= True )

model.save("model_GAIL")
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
plt.savefig('fig\GAIL-PPO二维仿真\Training_Progress.png')
plt.show()