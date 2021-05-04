import gym
import numpy as np
from copter_gym import Copter_Gym

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

env = Copter_Gym()
"""
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=100000)
model.save("ddpg_copter")

del model # remove to demonstrate saving and loading
"""
model = DDPG.load("ddpg_copter")

obs = env.reset()
while True:
    
    action, _states = model.predict(obs)
    print("Action:{}".format(action))
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    env.render()