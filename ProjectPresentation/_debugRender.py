import snake_gym
import gym
import os
from agent_pytorchGrid import Agent,ReplayBuffer,ReplayBufferGrid

import torch



if __name__ == '__main__':

    #get the environment
    env_name = "Snake-v0PyTorch"
    gs = (8,8)

    env = gym.make(env_name,env_config={"gs": gs,
                                              "BLOCK_SIZE": 20,
                                              "snake_length":0},)

    #iniialize model
    from GridAgent_SnakePytorchV0 import DQN
    # instantiate Q-network
    dqn = DQN(h= gs[0],
              w = gs[0],
              action_size=env.action_space.n,
              input_channels = env.reset().shape[0])
    dqn.load_state_dict(torch.load(r"Models/Snake-v0PyTorch_8x8/ffdqn_300000episodes.pth"))

    print("done")

