from MyEnvs import FighterEnv_nopolicy_2D
from print_fig import plot_and_save_fig

myenv = FighterEnv_nopolicy_2D(True,Dt = 0.01)


plot_and_save_fig(myenv, "2D", "fig/无突防二维仿真")
