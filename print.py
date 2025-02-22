from MyEnvs import FighterEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


myenv = FighterEnv(True)

total_reward = 0.0
model = PPO.load("model_1", env=myenv,device='cpu')

obs , _ = myenv.reset()
while myenv.check_terminated(obs) == False:
    action, _states = model.predict(obs)
    obs, rewards, dones, _ , _ = myenv.step(action)
    total_reward += rewards





print(obs)
print(myenv.FD.r)
print(total_reward)



plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["r"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["r"], label = "战斗机")
plt.xlabel('t/s')
plt.ylabel('r/m')
plt.legend()




plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["y"] ,label = "防御弹")
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["y"] , label = "战斗机")
plt.title('纵向轨迹图')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.legend()
plt.savefig('fig\PPO三维仿真\纵向轨迹图.png')

plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["z"],label = "防御弹")
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["z"], label = "战斗机")
plt.title('侧向轨迹图')
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.legend()
plt.savefig('fig\PPO三维仿真\侧向轨迹图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["theta"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["theta"], label = "战斗机")
plt.title('弹道倾角图')
plt.xlabel('t/s')
plt.ylabel('theta/rad')
plt.legend()
plt.savefig('fig\PPO三维仿真\弹道倾角图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["psi"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["psi"], label = "战斗机")
plt.title('弹道偏角图')
plt.xlabel('t/s')
plt.ylabel('psi/rad')
plt.legend()
plt.savefig('fig\PPO三维仿真\弹道偏角图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_y"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_y"], label = "战斗机")
plt.title('纵向加速度图')
plt.xlabel('t/s')
plt.ylabel('a_y/(m s^-2)')
plt.legend()
plt.savefig('fig\PPO三维仿真\纵向加速度图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_z"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_z"], label = "战斗机")
plt.title('侧向加速度图')
plt.xlabel('t/s')
plt.ylabel('a_y/(m s^-2)')
plt.legend()
plt.savefig('fig\PPO三维仿真\侧向加速度图.png')



plt.show()