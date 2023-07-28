import snake_gym
import gym
import os
from agent_pytorchGrid import Agent,ReplayBuffer,ReplayBufferGrid
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from agent_pytorchGrid import Agent
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use("ggplot_gundp")

import matplotlib
matplotlib.use("TkAgg")

class DQN(nn.Module):

    def __init__(self,  h, w, action_size = 4,input_channels = 2):
        super().__init__()

        self.action_size = action_size
        #
        # # Network
        # self.conv1 = nn.Conv2d(input_channels, 64, 3,stride=2)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64,32,3,stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.flat1 = nn.Flatten()
        # self.f1 = nn.LazyLinear(128)
        #
        # self.f2 = nn.LazyLinear(128)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.head = nn.Linear(128, self.action_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16*input_channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16*input_channels)
        self.conv3 = nn.Conv2d(16*input_channels, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, action_size)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """

        Args:
            x: Tensor representation of input states

        Returns:
            list of int: representing the Q values of each state-action pair
        """
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def render_video(env,agent, video_path,epsilon = 0.0,debug = False):

    env.metadata["render_fps"] = 15
    video = VideoRecorder(env, video_path)
    # returns an initial observation
    observation = env.reset()
    done = False
    while not done:
        env.render()
        video.capture_frame()
        # env.action_space.sample() produces either 0 (left) or 1 (right).

        #get action values
        state = torch.from_numpy(observation).unsqueeze(0).float()  # .float().to(self.device)
        agent.qnetwork_local.eval()
        with torch.no_grad():
            action_values = agent.qnetwork_local(state)

        # observation, reward, done, info = env.step(agent.act(observation,eps=epsilon,debug = debug))
        observation, reward, done, info = env.step(action_values.argmax().item())
        # action_values.argmax().item()
        # Not printing this time
        # print("step", i, observation, reward, done, info)
        if done:
            env.plot_state(actions = action_values)
            break

    video.close()
    env.close()

def simulate_epochs(env, agent,n_epochs = 100, epsilon=0.0,debug = True):

    list_score = []

    for i in tqdm(range(n_epochs)):
        observation = env.reset()
        done = False
        while not done:
            #get action values
            state = torch.from_numpy(observation).unsqueeze(0).float()  # .float().to(self.device)
            agent.qnetwork_local.eval()
            with torch.no_grad():
                action_values = agent.qnetwork_local(state)

            # observation, reward, done, info = env.step(agent.act(observation,eps=epsilon,debug = debug))
            observation, reward, done, info = env.step(action_values.argmax().item())

            if done:
                list_score.append((i,env.score))
                break
    return pd.DataFrame(list_score,columns=["Epochs","Score"])

def plot_history(df_out,n_rolling = 15):
    fig, ax = plt.subplots(1)
    sns.lineplot(x="Epochs", y="Score", data=df_out)
    ax.plot(df_out["Score"].rolling(n_rolling).mean())
    ax.axhline(df_out.Score.mean(), color="red", linewidth=1, linestyle="--")
    plt.tight_layout()
    ax.set_ylim(0,80)
    fig.savefig("Results_Rolling_{}.svg".format(n_rolling))
    return fig,ax

if __name__ == '__main__':

    #get the environment
    env_name = "Snake-v0PyTorch"
    gs = (20,20)

    env = gym.make(env_name,env_config={"gs": gs,
                                              "BLOCK_SIZE": 20,
                                              "snake_length":0},)

    #iniialize model

    # instantiate Q-network
    dqn = DQN(h= gs[0],
              w = gs[0],
              action_size=env.action_space.n,
              input_channels = env.reset().shape[0])
    dqn.load_state_dict(torch.load(r"./logs/dqn_Snake-v0PyTorch_20x20/20221212-210342/ffdqn_20000episodes.pth"))

    agent = Agent(dqn,
                  None,
                  lr=1e-4,
                  batch_size=128,
                  update_every=10,
                  gamma=0.99,
                  tau=.1,
                  epsilon=.1,
                  epsilon_decay=.99,
                  render=True,
                  optimal=True,
                  )

    df_out = simulate_epochs(env,agent,n_epochs=1000,epsilon = 0.0,debug = False)

    fig,ax = plot_history(df_out,n_rolling = 20)





    render_video(env, agent, r"SnakeV1_Final20000_2.mp4", epsilon=0.0,debug = True)


    print("done")

