import gym
from gym.utils.play import play
import pygame
import snake_gymDirected

keys_to_action ={
                                                       "s": 0,
                                                       "d": 1,
                                                       "a": 2,


                                                      }
play(gym.make("SnakeDir-v0",
              env_config = dict({
                "gs":(10,10),
                "BLOCK_SIZE":20,
              })),
     keys_to_action=keys_to_action,noop = 0,
     fps = 10
     )