from stable_baselines.common.env_checker import check_env
from copter_gym import Copter_Gym

env=Copter_Gym()

check_env(env)

print("*******Done********")

