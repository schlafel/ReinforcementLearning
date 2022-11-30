
import os


import tensorflow as tf
import gym
import snake_gym
import numpy as np
import tqdm




if __name__ == '__main__':

    env = gym.make("Snake-v0",env_config = {"w":400,
                                             "h":400,
                                            "BLOCK_SIZE":20},
                   )


    NUM_ACTIONS = env.action_space.n-1
    NUM_STATES = env.observation_space.n
    Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #You could also make this dynamic if you don't know all games states upfront
    gamma = 0.9 # discount factor
    alpha = 0.9 # learning rate
    for episode in range(1,1001):
        done = False
        rew_tot = 0
        obs = env.reset()
        while done != True:
                action = np.argmax(Q[obs]) #choosing the action with the highest Q value
                obs2, rew, done, info = env.step(action) #take the action
                Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation
                #Q[obs,action] = rew + gamma * np.max(Q[obs2]) # same equation but with learning rate = 1 returns the basic Bellman equation
                rew_tot = rew_tot + rew
                obs = obs2
        if episode % 50 == 0:
            print('Episode {} Total Reward: {}'.format(episode,rew_tot))