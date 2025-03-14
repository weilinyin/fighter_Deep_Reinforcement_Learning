from MyEnvs import FighterEnv
import matplotlib.pyplot as plt

def plot_and_save_fig(env:FighterEnv, mode , save_path:str):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    
    plt.figure()
    plt.plot(env.t_array , env.plotdata["defender"]["r"],label = "R_FD")
    plt.plot(env.t_array , env.plotdata["fighter"]["r"], label = "R_FT")
    plt.title('相对距离图')
    plt.xlabel('t/s')
    plt.ylabel('r/m')
    plt.legend()
    plt.savefig(save_path + "/相对距离图.png")

    if mode == "3D":
        # 创建一个新的figure
        fig = plt.figure()

        # 添加一个3D子图
        ax = fig.add_subplot(111, projection='3d')


        # 绘制线图
        ax.plot(env.plotdata["defender"]["x"], env.plotdata["defender"]["y"], env.plotdata["defender"]["z"], label='防御弹')
        ax.plot(env.plotdata["fighter"]["x"], env.plotdata["fighter"]["y"], env.plotdata["fighter"]["z"], label='战斗机')

        # 设置标签
        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.set_zlabel('z/m')
        ax.legend()

        plt.figure()
        plt.plot(env.plotdata["defender"]["x"] , env.plotdata["defender"]["y"] ,label = "防御弹")
        plt.plot(env.plotdata["fighter"]["x"] , env.plotdata["fighter"]["y"] , label = "战斗机")
        plt.title('纵向轨迹图')
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.legend()
        plt.savefig(save_path + "\纵向轨迹图.png")
        plt.figure()
        plt.plot(env.t_array , env.plotdata["defender"]["theta"],label = "防御弹")
        plt.plot(env.t_array , env.plotdata["fighter"]["theta"], label = "战斗机")
        plt.title('弹道倾角图')
        plt.xlabel('t/s')
        plt.ylabel('theta/rad')
        plt.legend()
        plt.savefig(save_path + "\弹道倾角图.png")
        plt.figure()
        plt.plot(env.t_array , env.plotdata["defender"]["a_y"],label = "防御弹")
        plt.plot(env.t_array , env.plotdata["fighter"]["a_y"], label = "战斗机")
        plt.title('纵向加速度图')
        plt.xlabel('t/s')
        plt.ylabel('a_y/(m s^-2)')
        plt.legend()
        plt.savefig(save_path + "\纵向加速度图.png")



    plt.figure()
    plt.plot(env.plotdata["defender"]["x"] , env.plotdata["defender"]["z"],label = "防御弹")
    plt.plot(env.plotdata["fighter"]["x"] , env.plotdata["fighter"]["z"], label = "战斗机")
    plt.title('侧向轨迹图')
    plt.xlabel('x/m')
    plt.ylabel('z/m')
    plt.legend()
    plt.savefig(save_path + "\侧向轨迹图.png")


    plt.figure()
    plt.plot(env.t_array , env.plotdata["defender"]["psi"],label = "防御弹")
    plt.plot(env.t_array , env.plotdata["fighter"]["psi"], label = "战斗机")
    plt.title('弹道偏角图')
    plt.xlabel('t/s')
    plt.ylabel('psi/rad')
    plt.legend()
    plt.savefig(save_path + "\弹道偏角图.png")


    plt.figure()
    plt.plot(env.t_array , env.plotdata["defender"]["a_z"],label = "防御弹")
    plt.plot(env.t_array , env.plotdata["fighter"]["a_z"], label = "战斗机")
    plt.title('侧向加速度图')
    plt.xlabel('t/s')
    plt.ylabel('a_z/(m s^-2)')
    plt.legend()
    plt.savefig(save_path + "\侧向加速度图.png")



    plt.show()

        

    
