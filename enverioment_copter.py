import numpy as np
from utils import Utils
from quadcopter_model import dynamic_model
import math
u = Utils()


class Enverioment:
    def __init__(self):
        self.T = 1
        self.dtau = 0.01
        self.dt = 0.001

        self.index = 0

        self.action_space=4
        self.action_space_min=-1
        self.action_space_max=1
        self.obs_space=6

        self.N = int(self.T / self.dtau)
        self.Nsolver = int(self.dtau / self.dt)
        
        self.states=np.zeros(12)

        self.m = 0.65
        self.g = 9.81
        self.f_initial = self.m * self.g

        self.obs = self.__obs_calc()

    def reset(self):
            self.states=np.zeros(12)
            self.index=0
            return self.__obs_calc()

    def __obs_calc(self):
            phi, theta = self.states[6:8]
            u, v, w = self.states[3:6]
            r=self.states[11]
            return [u,v,w,phi,theta,r]

    def step(self,action):
            action=self.__action_converter(action)
            self.states=dynamic_model(self.states,action,self.Nsolver,self.dt)
            self.obs=self.__obs_calc()
            reward=self.__reward_calc()
            done,reward_extra=self.__done_calc()
            self.index+=1
            return self.obs,reward+reward_extra,done,{}
    def __action_converter(self,action):
            for i in range(4):
              action[i]=action[i]
            action_f=action[3]
            action_f=self.f_initial+action_f*self.f_initial/2
            action[3]=action_f
      
            return action

    def __reward_calc(self):
            reward=0
            u,v,w,phi,theta,r=self.obs
            V=math.sqrt(pow(u,2)+pow(v,2)+pow(w,2))
            W=pow(phi,2)+pow(theta,2)+pow(r,2)
            reward=-1*(V+W)
            return reward

    def __done_calc(self):
            if self.index>=(self.N-1):
                return [True,0]
            phi, theta = self.states[6:8]
            r=self.states[11]
            if abs(phi)>math.pi/2 or abs(theta) >math.pi/2 or abs(r)>math.pi/2:
                return [True,-1000000]
            return [False,0]