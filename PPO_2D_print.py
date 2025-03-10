from MyEnvs import FighterEnv_2D
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


myenv = FighterEnv_2D(True,Dt = 0.01 ,dt = 0.01)


model = PPO.load("model_2D", env=myenv,device='cpu')

obs , _ = myenv.reset()
while not (myenv.success or myenv.fail):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ , _ = myenv.step(action)





print(obs)
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
plt.savefig('fig\PPO二维仿真\相对距离图.png')





plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["z"],label = "防御弹")
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["z"], label = "战斗机")
plt.title('侧向轨迹图')
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.legend()
plt.savefig('fig\PPO二维仿真\侧向轨迹图.png')


plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["psi"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["psi"], label = "战斗机")
plt.title('弹道偏角图')
plt.xlabel('t/s')
plt.ylabel('psi/rad')
plt.legend()
plt.savefig('fig\PPO二维仿真\弹道偏角图.png')


plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_z"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_z"], label = "战斗机")
plt.title('侧向加速度图')
plt.xlabel('t/s')
plt.ylabel('a_z/(m s^-2)')
plt.legend()
plt.savefig('fig\PPO二维仿真\侧向加速度图.png')



plt.show()