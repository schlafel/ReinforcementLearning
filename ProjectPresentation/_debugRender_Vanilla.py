import snake_gym
import pandas as pd
import gym
import os
from agent_pytorchGrid import Agent,ReplayBuffer,ReplayBufferGrid
from VanillaAgent_SnakePyTorch import DQN
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from tqdm import tqdm
style.use("ggplot_gundp")

import matplotlib
matplotlib.use("TkAgg")

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from agent_pytorchGrid import Agent


def render_video(env,agent, video_path,epsilon = 0.0,debug = False):


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

        pd.DataFrame([observation], columns=["Danger straight",
                                             "Danger right",
                                             "Danger left",
                                             "Move direction left",
                                             "Move direction right",
                                             "Move direction up",
                                             "Move direction down",
                                             "Food to the left",
                                             "Food to the right",
                                             "Food upwards",
                                             "Food downwards",
                                             ]).T
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
    fig.savefig("Results_RollingVanilla_{}.svg".format(n_rolling))
    return fig,ax

if __name__ == '__main__':

    #get the environment
    env_name = "Snake-Vanilla"
    gs = (20,20)

    env = gym.make(env_name,env_config={"gs": gs,
                                              "BLOCK_SIZE": 20,
                                              "snake_length":0},)
    env.metadata["render_fps"] = 20
    #iniialize model

    # instantiate Q-network
    dqn = DQN(action_size=env.action_space.n,
              input_dim=env.observation_space.n)

    dqn.load_state_dict(torch.load(r".\Models\Snake-Vanilla_20x20\ffdqn_500episodes.pth"))

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



    render_video(env, agent, r"VanillaSnake.mp4", epsilon=0.0,debug = True)


    print("done")

