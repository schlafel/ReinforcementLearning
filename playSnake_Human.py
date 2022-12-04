import gym
from gym.utils.play import play
import pygame
import snake_gym

keys_to_action ={
                   "d": 0,
                   "s": 1,
                   "a": 2,
                   "w": 3,
                  }

env_config = dict({"gs": (20, 20),
                               "BLOCK_SIZE":20,
                               "snake_length":0})

play(gym.make("Snake-Vanilla",env_config=env_config),
     keys_to_action=keys_to_action,noop = 1,
     fps = 10,

     )