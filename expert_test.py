from MyEnvs import FighterEnv_2D
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from GAIL_PPO import expert_generator
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


myenv = FighterEnv_2D(True,Dt = 0.01 ,dt = 0.01)



target_c_f = 0
c_f = -0.0015

obs , _ = myenv.reset()
    
expert = expert_generator(myenv , c_f)
    
while not (myenv.success or myenv.fail):
    a_E = expert.generate(myenv.t)
    action = np.array([a_E])
    obs, rewards, dones, _ , _ = myenv.step(action)




print(rewards)
r = myenv.plotdata["defender"]["r"]
print(min(r))

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["r"],label = "R_FD")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["r"], label = "R_FT")
plt.title('相对距离图')
plt.xlabel('t/s')
plt.ylabel('r/m')
plt.legend()






plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["z"],label = "防御弹")
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["z"], label = "战斗机")
plt.title('侧向轨迹图')
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.legend()



plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["psi"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["psi"], label = "战斗机")
plt.title('弹道偏角图')
plt.xlabel('t/s')
plt.ylabel('psi/rad')
plt.legend()



plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_z"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_z"], label = "战斗机")
plt.title('侧向加速度图')
plt.xlabel('t/s')
plt.ylabel('a_z/(m s^-2)')
plt.legend()




plt.show()