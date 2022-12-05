import snake_gym
import gym
import os
from agent_pytorchGrid import Agent,ReplayBuffer,ReplayBufferGrid

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary

class DQN(nn.Module):

    def __init__(self, action_size, input_channels=1):
        super().__init__()

        self.action_size = action_size

        # Network
        self.conv1 = nn.Conv2d(input_channels, 64, 3,stride=2)
        self.conv2 = nn.Conv2d(64,32,3,stride=2)

        self.flat1 = nn.Flatten()
        self.f1 = nn.LazyLinear(128)

        self.f2 = nn.LazyLinear(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.head = nn.Linear(128, self.action_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat1(x)

        #print(x.shape)
        x = F.relu((self.f1(x)))
        x = F.relu(self.bn2(self.f2(x)))

        return self.head(x)



def DeepQLearning(env: gym.Env, agent: object, num_episodes: int, max_steps=1000, save_model=None,render = False):
    reward_per_ep = list()

    for i in tqdm(range(num_episodes)):
        reward = agent.episode(env, max_steps=max_steps)
        reward_per_ep.append(reward)

    if save_model is not None:
        torch.save(agent.qnetwork_local.state_dict(), save_model)

    return reward_per_ep



if __name__ == '__main__':
    # set parameters of the Agent and ReplayBuffer
    lr = 1e-4
    batch_size = 64
    update_every = 5
    gamma = 0.99
    tau = 0.5
    epsilon = 0.1


    # number of episodes and file path to save the model
    num_episodes = 1500

    model_dir = os.path.join(".", 'Models')
    save_model = os.path.join(model_dir, 'ffdqn_{}episodes.pth'.format(num_episodes))
    os.makedirs(model_dir, exist_ok=True)

    buffer_size = int(1e+3)
    seed = 0
    render = True
    env = gym.make("Snake-v2PyTorch",env_config={"gs": (10, 10),
                                              "BLOCK_SIZE": 20,
                                              "snake_length":0},)

    # instantiate Q-network
    dqn = DQN(action_size=env.action_space.n,
              input_channels = env.reset().shape[0])
    summary(dqn.cuda(), input_size=env.reset().shape)
    # instantiate memory buffer
    env.reset()
    env.plot_state()
    memory = ReplayBufferGrid(obs_dim=env.reset().shape,
                          size=buffer_size,
                          batch_size=batch_size)
    # instantiate agent
    agent = Agent(dqn,
                  memory,
                  lr=lr,
                  batch_size=batch_size,
                  update_every=update_every,
                  gamma=gamma,
                  tau=tau,
                  epsilon=epsilon,
                  render = render)


    R = DeepQLearning(env, agent, num_episodes, save_model=save_model)