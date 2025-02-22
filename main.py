from MyEnvs import FighterEnv
from stable_baselines3 import PPO


myenv = FighterEnv()

model_1 = PPO("MlpPolicy", myenv, verbose=1 , device='cpu')
model_1.learn(total_timesteps=100000,log_interval = 4)
model_1.save("model_1")
del model_1

