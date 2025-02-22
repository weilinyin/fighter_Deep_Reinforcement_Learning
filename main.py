from MyEnvs import FighterEnv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


myenv = FighterEnv()

model_1 = PPO("MlpPolicy", myenv, verbose=1 , device='cpu' ,
              learning_rate=0.002 ,gamma=0.97 , n_steps=2048, clip_range=0.2)
model_1.learn(total_timesteps=100000)
model_1.save("model_1")
del model_1

