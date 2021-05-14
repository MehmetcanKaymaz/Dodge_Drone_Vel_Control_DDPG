import gym
import numpy as np
from copter_gym import Copter_Gym

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import FeedForwardPolicy

class CustomDDPGPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[64, 64, 64],
                                           layer_norm=False,
                                           feature_extraction="mlp")
                                           
class CustomDDPGPolicy_2(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[128, 128, 128],
                                           layer_norm=False,
                                           feature_extraction="mlp")                                           

env = Copter_Gym()

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
action_noise_2 = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(1) * np.ones(n_actions))

model = DDPG(CustomDDPGPolicy_2, env, verbose=1, param_noise=param_noise, action_noise=action_noise,gamma=0.9)
model_2 = DDPG(CustomDDPGPolicy_2, env, verbose=1, param_noise=param_noise, action_noise=action_noise,gamma=0.95)
model_3 = DDPG(CustomDDPGPolicy_2, env, verbose=1, param_noise=param_noise, action_noise=action_noise,gamma=0.99)
model_4 = DDPG(CustomDDPGPolicy_2, env, verbose=1, param_noise=param_noise, action_noise=action_noise,gamma=0.995)

print("Model 1 Train ")
model.learn(total_timesteps=100000)
print("Model 2 Train ")
model_2.learn(total_timesteps=100000)
print("Model 3 Train ")
model_3.learn(total_timesteps=100000)
print("Model 4 Train ")
model_4.learn(total_timesteps=100000)

model.save("ddpg_copter")
model_2.save("ddpg_copter_2")
model_2.save("ddpg_copter_3")
model_4.save("ddpg_copter_4")

del model # remove to demonstrate saving and loading
del model_2
del model_3
del model_4


print("*************************************\n        Model 1 Result        \n*************************************")

model = DDPG.load("ddpg_copter")
n_episode=10
episode_reward=np.zeros(n_episode)
for i in range(n_episode):
    obs = env.reset()
    sum_reward=0
    while True:
    
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        sum_reward+=rewards
        if dones:
            break
        env.render()
    episode_reward[i]=sum_reward

model_1_result=np.mean(episode_reward)
    
print("*************************************\n        Model 2 Result        \n*************************************")

model_2 = DDPG.load("ddpg_copter_2")
n_episode=10
episode_reward=np.zeros(n_episode)
for i in range(n_episode):
    obs = env.reset()
    sum_reward=0
    while True:
    
        action, _states = model_2.predict(obs)
        obs, rewards, dones, info = env.step(action)
        sum_reward+=rewards
        if dones:
            break
        env.render()
    episode_reward[i]=sum_reward

model_2_result=np.mean(episode_reward)
        
print("*************************************\n        Model 3 Result        \n*************************************")

model_3 = DDPG.load("ddpg_copter_3")

n_episode=10
episode_reward=np.zeros(n_episode)
for i in range(n_episode):
    obs = env.reset()
    sum_reward=0
    while True:
    
        action, _states = model_3.predict(obs)
        obs, rewards, dones, info = env.step(action)
        sum_reward+=rewards
        if dones:
            break
        env.render()
    episode_reward[i]=sum_reward

model_3_result=np.mean(episode_reward)

print("*************************************\n        Model 4 Result        \n*************************************")

model_4 = DDPG.load("ddpg_copter_4")

n_episode=10
episode_reward=np.zeros(n_episode)
for i in range(n_episode):
    obs = env.reset()
    sum_reward=0
    while True:
    
        action, _states = model_4.predict(obs)
        obs, rewards, dones, info = env.step(action)
        sum_reward+=rewards
        if dones:
            break
        env.render()
    episode_reward[i]=sum_reward

model_4_result=np.mean(episode_reward)


print("********************\n        Results        \n******************")
print("Model 1(gamma=0.9) : {} avarage reward".format(model_1_result))
print("Model 2(gamma=0.95) : {} avarage reward".format(model_2_result))
print("Model 3(gamma=0.99) : {} avarage reward".format(model_3_result))
print("Model 4(gamma=0.995) : {} avarage reward".format(model_4_result))  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
