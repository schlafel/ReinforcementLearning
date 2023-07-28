import snake_gym
import gym
import os
from agent_pytorch import Agent,ReplayBuffer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tensorflow as tf
import datetime
from tqdm import tqdm
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torchsummary import summary
def render_video(env,agent, video_path,epsilon = 0.0):

    env.metadata["render_fps"] = 5
    video = VideoRecorder(env, video_path)
    # returns an initial observation
    observation = env.reset()
    done = False
    while not done:
        env.render()
        video.capture_frame()
        # env.action_space.sample() produces either 0 (left) or 1 (right).
        observation, reward, done, info = env.step(agent.act(observation,eps=epsilon))
        # Not printing this time
        # print("step", i, observation, reward, done, info)
        if done:
            break
    video.close()
    env.close()


class DQN(nn.Module):

    def __init__(self, action_size, input_dim=1):
        super().__init__()

        self.action_size = action_size

        # Network
        self.f1 = nn.Linear(input_dim, 128)

        self.f2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.head = nn.Linear(128, self.action_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu((self.f1(x)))
        x = F.relu(self.bn2(self.f2(x)))

        return self.head(x)



def DeepQLearning(env: gym.Env, agent: Agent, num_episodes: int, max_steps=1000,
                  save_model=None,
                  env_name = "",
                  render = False):


    reward_per_ep = list()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn_{}/'.format(os.path.basename(os.path.dirname(save_model))) + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    for i in tqdm(range(num_episodes)):
        score,n_steps, losses = agent.episode(env, max_steps=max_steps)
        # reward_per_ep.append(reward)
        with summary_writer.as_default():
            tf.summary.scalar('Score', env.score, step=i)
            tf.summary.scalar('High-Score', env.high_score, step=i)
            tf.summary.scalar('Number of Play Steps', n_steps, step=i)
            tf.summary.scalar('Losses', np.nanmean(losses), step=i)
        # reward_per_ep.append(reward)
        if (i%100 == 0) & (i != 0):
            render_video(env,agent,
                         os.path.join(log_dir,
                                      "Video_trained_{:d}.mp4".format(i)))

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
    num_episodes = 500


    buffer_size = int(1e+3)
    seed = 0
    render = False
    env_name ="Snake-Vanilla"
    gs = (20, 20)


    env = gym.make(env_name,env_config={"gs": gs,
                                              "BLOCK_SIZE": 20,
                                              "snake_length":0},)

    env.metadata["render_fps"] = 15


    model_dir = os.path.join(".", 'Models',env_name + "_" + "x".join([str(g) for g in gs]))
    save_model = os.path.join(model_dir, 'ffdqn_{}episodes.pth'.format(num_episodes))
    os.makedirs(model_dir, exist_ok=True)

    # instantiate Q-network
    dqn = DQN(action_size=env.action_space.n,
              input_dim=env.observation_space.n)

    summary(dqn.cuda(), input_size=env.reset().shape)
    # instantiate memory buffer
    memory = ReplayBuffer(obs_dim=env.observation_space.n,
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


    R = DeepQLearning(env, agent, num_episodes, save_model=save_model,
                      env_name=env_name)