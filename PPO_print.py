from MyEnvs import FighterEnv
from stable_baselines3 import PPO
from print_fig import plot_and_save_fig




myenv = FighterEnv(True)


model = PPO.load("model_1", env=myenv,device='cpu')

obs , _ = myenv.reset()
while not (myenv.success or myenv.fail):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ , _ = myenv.step(action)





print(obs)
print(rewards)


plot_and_save_fig(myenv, "3D", "fig/PPO三维仿真")