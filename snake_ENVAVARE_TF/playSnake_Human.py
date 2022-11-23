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
play(gym.make("Snake-v0"),
     keys_to_action=keys_to_action,noop = 4,
     fps = 10
     )