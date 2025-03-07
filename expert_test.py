from MyEnvs import FighterEnv_2D
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from GAIL_PPO import expert_generator
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


myenv = FighterEnv_2D(False,Dt = 0.01 ,dt = 0.01)



target_c_f = 0
c_f = 0.1
while c_f > -0.1:
    obs , _ = myenv.reset()
    
    expert = expert_generator(myenv.FD , myenv.FT , myenv.t_0 , c_f )
    
    while not (myenv.success or myenv.fail):
        a_E = expert.generate(myenv.t)
        action = np.array([a_E])
        obs, rewards, dones, _ , _ = myenv.step(action)
    
    
    if rewards >30:
        target_c_f = c_f
        print(target_c_f)
        break
    
    c_f -= 0.001


print(target_c_f)




print(rewards)
