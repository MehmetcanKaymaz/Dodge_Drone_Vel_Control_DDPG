import gym
from gym import spaces
from gym.utils import seeding
from enverioment_copter import Enverioment
import numpy as np
import math
from utils import Utils

ut=Utils()


class Copter_Gym(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(Copter_Gym, self).__init__()
    self.env=Enverioment()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    
    self.min_action=self.env.action_space_min
    self.max_action=self.env.action_space_max
    self.action_space = spaces.Box(
        low=self.min_action,
        high=self.max_action,
        shape=(4,),
        dtype=np.float32
    )    
    
    # Example for using image as input:
    self.observation_space = spaces.Box(low=-10, high=10,
                                        shape=(6,), dtype=np.float32)
    
    self.seed()
    self.observation=self.reset()
    

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  

  def step(self, action):
    self.observation, reward, done, info=self.env.step(action)
    self.observation=np.array(self.observation)
    return self.observation, reward, done, info
  def reset(self):
    observation=self.env.reset()
    observation=np.array(observation)
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    u,v,w,phi,theta,r=self.observation
    print("u:{}(m/s)  v:{}(m/s)  w:{}(m/s)  phi:{}(deg)  theta:{}(deg)  r:{}(deg/sec)".
          format(u,v,w,ut.rad_to_deg(phi),ut.rad_to_deg(theta),ut.rad_to_deg(r)))
  def close (self):
    print("********** Done ************")