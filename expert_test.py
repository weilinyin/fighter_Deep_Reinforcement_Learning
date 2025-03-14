from MyEnvs import FighterEnv_2D
from GAIL_PPO import expert_generator
import numpy as np
from print_fig import plot_and_save_fig


myenv = FighterEnv_2D(True,Dt = 0.01 ,dt = 0.01)



target_c_f = 0
c_f = -0.002

obs , _ = myenv.reset()
    
expert = expert_generator(myenv , c_f)
    
while not (myenv.success or myenv.fail):
    a_E = expert.generate(myenv.t)
    action = np.array([a_E])
    obs, rewards, dones, _ , _ = myenv.step(action)




print(rewards)
plot_and_save_fig(myenv, "2D", "fig/GAIL-PPO仿真/专家策略")

