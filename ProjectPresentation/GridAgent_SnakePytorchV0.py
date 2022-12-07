import snake_gym
import gym
import os
from agent_pytorchGrid import Agent,ReplayBuffer,ReplayBufferGrid

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary
import tensorflow as tf
import datetime
from gym.wrappers.monitoring.video_recorder import VideoRecorder
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



def DeepQLearning(env: gym.Env, agent: Agent, num_episodes: int,env_name:str, max_steps=1000,
                  save_model=None,
                  render = False):
    reward_per_ep = list()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn_{}/'.format(env_name) + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    for i in tqdm(range(num_episodes)):
        reward,n_steps = agent.episode(env, max_steps=max_steps)
        with summary_writer.as_default():
            tf.summary.scalar('Score', env.score, step=i)
            tf.summary.scalar('High-Score', env.high_score, step=i)
            tf.summary.scalar('Number of Play Steps', n_steps, step=i)
        reward_per_ep.append(reward)
        if i%1000 == 0:
            render_video(env,agent,
                         os.path.join(log_dir,
                                      "Video_trained_{:d}.mp4".format(i)))


    if save_model is not None:
        # torch.save(agent.qnetwork_local.state_dict(), save_model)
        torch.save(agent.qnetwork_local.state_dict(), os.path.join(log_dir,os.path.basename(save_model)))

    return reward_per_ep,log_dir


def render_video(env,agent, video_path,epsilon = 0.0):

    env.metadata["render_fps"] = 5
    video = VideoRecorder(env, video_path)
    # returns an initial observation
    observation = env.reset()
    for i in range(0,1000):
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




if __name__ == '__main__':
    # set parameters of the Agent and ReplayBuffer
    lr = 1e-4
    batch_size = 256
    update_every = 10
    gamma = 0.99
    tau = 0.5
    epsilon = 0.1


    # number of episodes and file path to save the model
    num_episodes = 20_000



    buffer_size = int(1e+5)
    seed = 0
    render = False
    env_name = "Snake-v0PyTorch"
    env = gym.make(env_name,env_config={"gs": (10, 10),
                                              "BLOCK_SIZE": 20,
                                              "snake_length":1},)


    model_dir = os.path.join(".", 'Models',env_name)
    save_model = os.path.join(model_dir, 'ffdqn_{}episodes.pth'.format(num_episodes))
    os.makedirs(model_dir, exist_ok=True)

    # instantiate Q-network
    dqn = DQN(action_size=env.action_space.n,
              input_channels = env.reset().shape[0])
    summary(dqn.cuda(), input_size=env.reset().shape)
    # instantiate memory buffer
    env.reset()
    #env.plot_state()
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
                  render = render,
                  optimal = False)


    R,log_dir = DeepQLearning(env, agent, num_episodes, save_model=save_model,
                      env_name=env_name)










