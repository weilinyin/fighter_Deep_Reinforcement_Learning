from MyEnvs import FighterEnv_2D
from stable_baselines3 import PPO
from print_fig import plot_and_save_fig



myenv = FighterEnv_2D(True,Dt = 0.01 ,dt = 0.01)


model = PPO.load("model_GAIL", env=myenv,device='cpu')

obs , _ = myenv.reset()
while not (myenv.success or myenv.fail):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ , _ = myenv.step(action)

plot_and_save_fig(myenv, "2D", "fig/GAIL-PPO仿真")



