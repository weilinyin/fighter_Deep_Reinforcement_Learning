from MyEnvs import FighterEnv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


myenv = FighterEnv()

model_1 = PPO("MlpPolicy", myenv, verbose=1 , device='cpu')
model_1.learn(total_timesteps=10000)

model_1.save("model_1")
del model_1

