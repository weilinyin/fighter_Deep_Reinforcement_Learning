from MyEnvs import FighterEnv
import matplotlib.pyplot as plt
import math as m
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

myenv = FighterEnv(True)

obs , _ = myenv.reset()
for i in range(1000):
    a_y , a_z = myenv.FT.proportional_navigation()
    action = np.array([m.asin(a_z/m.sqrt(a_y **2 + a_z **2)) , m.sqrt(a_y **2 + a_z **2)])
    obs, rewards, dones, _ , _ = myenv.step(action)









plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["y"] ,label = "防御弹")
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["y"] , label = "战斗机")
plt.title('纵向轨迹图')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.legend()
plt.savefig('fig\无突防三维仿真\纵向轨迹图.png')

plt.figure()
plt.plot(myenv.plotdata["defender"]["x"] , myenv.plotdata["defender"]["z"],label = "防御弹")
plt.plot(myenv.plotdata["fighter"]["x"] , myenv.plotdata["fighter"]["z"], label = "战斗机")
plt.title('侧向轨迹图')
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.legend()
plt.savefig('fig\无突防三维仿真\侧向轨迹图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["theta"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["theta"], label = "战斗机")
plt.title('弹道倾角图')
plt.xlabel('t/s')
plt.ylabel('theta/rad')
plt.legend()
plt.savefig('fig\无突防三维仿真\弹道倾角图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["psi"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["psi"], label = "战斗机")
plt.title('弹道偏角图')
plt.xlabel('t/s')
plt.ylabel('psi/rad')
plt.legend()
plt.savefig('fig\无突防三维仿真\弹道偏角图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_y"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_y"], label = "战斗机")
plt.title('纵向加速度图')
plt.xlabel('t/s')
plt.ylabel('a_y/(m s^-2)')
plt.legend()
plt.savefig('fig\无突防三维仿真\纵向加速度图.png')

plt.figure()
plt.plot(myenv.t_array , myenv.plotdata["defender"]["a_z"],label = "防御弹")
plt.plot(myenv.t_array , myenv.plotdata["fighter"]["a_z"], label = "战斗机")
plt.title('侧向加速度图')
plt.xlabel('t/s')
plt.ylabel('a_y/(m s^-2)')
plt.legend()
plt.savefig('fig\无突防三维仿真\侧向加速度图.png')



plt.show()