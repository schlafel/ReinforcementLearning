import gym
from gym.utils.play import play
import pygame
import numpy as np

# env = gym.make("MountainCar-v0")
# state = env.reset()


keys_to_action ={
                                                       "a": 0,
                                                       "s": 1,
                                                       "d": 2,

                                                      }


play( gym.make("MountainCar-v0"), keys_to_action=keys_to_action, noop=0)



#
# done = False
# while not done:
#     action = 2  # always go right!
#     new_state,render,_,_ = env.step(action)
#     print(new_state, render)
#     env.render(mode="human")
#
# env.close()