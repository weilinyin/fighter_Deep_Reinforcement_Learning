from MyEnvs import FighterEnv
from stable_baselines3 import PPO



myenv = FighterEnv()

model_1 = PPO("MlpPolicy", myenv, verbose=1, device='cpu',learning_rate = 0.005,
              gae_lambda= 0.98 , gamma = 0.96 , n_steps = 2048 , batch_size = 256 , n_epochs = 4 ,clip_range = 0.2 )
model_1.learn(total_timesteps=100000, log_interval=4)
model_1.save("model_1")
del model_1
