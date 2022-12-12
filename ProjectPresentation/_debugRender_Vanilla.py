import snake_gym
import pandas as pd
import gym
import os
from agent_pytorchGrid import Agent,ReplayBuffer,ReplayBufferGrid
from VanillaAgent_SnakePyTorch import DQN
import torch
from torch import nn
import torch.nn.functional as F

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

    render_video(env, agent, r"VanillaSnake.mp4", epsilon=0.0,debug = True)


    print("done")

