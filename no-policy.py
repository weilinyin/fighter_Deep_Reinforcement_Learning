from MyEnvs import FighterEnv_nopolicy
import numpy as np
from print_fig import plot_and_save_fig


myenv = FighterEnv_nopolicy(True , dt=0.01 ,Dt = 0.01)


r = np.array(myenv.plotdata["defender"]["r"])
print(min(r))


plot_and_save_fig(myenv, "2D", "fig/无突防三维仿真")