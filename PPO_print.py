from MyEnvs import FighterEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


myenv = FighterEnv(True)


model = PPO.load("model_1", env=myenv,device='cpu')

obs , _ = myenv.reset()
while not (myenv.success or myenv.fail):
    action, _states = model.predict(obs)
    obs, rewards, dones, _ , _ = myenv.step(action, deterministic=True)





print(obs)
print(rewards)


plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["r"],label = "R_FD")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["r"], label = "R_FT")
plt.title('相对距离图')
plt.xlabel('t/s')
plt.ylabel('r/m')
plt.legend()
plt.savefig('fig\PPO三维仿真\相对距离图.png')

# 创建一个新的figure
fig = plt.figure()

# 添加一个3D子图
ax = fig.add_subplot(111, projection='3d')


# 绘制线图
ax.plot(myenv.plotdata["defender"]["x"], myenv.plotdata["defender"]["y"], myenv.plotdata["defender"]["z"], label='防御弹')
ax.plot(myenv.plotdata["fighter"]["x"], myenv.plotdata["fighter"]["y"], myenv.plotdata["fighter"]["z"], label='战斗机')



# 设置标签
ax.set_xlabel('x/m')
ax.set_ylabel('y/m')
ax.set_zlabel('z/m')
ax.legend()





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