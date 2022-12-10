from math import tau
import torch
from torch import optim
import random
import numpy as np
import tensorflow as tf

import torch.nn.functional as F
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):

        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        # idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        idxs = np.random.randint(0,self.size, size=self.batch_size, )

        return (torch.from_numpy(self.obs_buf[idxs]).to(self.device),
                torch.from_numpy(self.acts_buf[idxs]).long().to(self.device),
                torch.from_numpy(self.rews_buf[idxs]).to(self.device),
                torch.from_numpy(self.next_obs_buf[idxs]).to(self.device),
                torch.from_numpy(self.done_buf[idxs]).to(self.device))

    def __len__(self):
        return self.size


class ReplayBufferGrid(ReplayBuffer):
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        super(ReplayBufferGrid, self).__init__(obs_dim = 1, size = 1,batch_size = batch_size)

        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, dqn, memory,
                 lr=1e-4,
                 batch_size=64,
                 update_every=5,
                 gamma=0.99, tau=1e-3,
                 epsilon=0.1,
                 epsilon_decay = 0.99,
                 seed=0,
                 render=False, optimal=False):
        """Initialize an Agent object.

        Params
        ======
            dqn (nn.Module): Module implementing the DQN
            memory (object): Replay buffer object
            seed (int): Random seed
        """

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network
        self.qnetwork_local = dqn.to(self.device)
        self.qnetwork_target = dqn.to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Other params
        self.lr = lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.render = render
        self.optimal = optimal

        # Replay memory
        self.memory = memory
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

    def episode(self, env, max_steps=1000,episode = 1):

        state = env.reset()
        score = 0
        _ = 1
        loss_list = []
        while True:
            if self.optimal:
                eps = 0
            else:
                eps = max(0.01, self.epsilon * self.epsilon_decay**episode)


            action = self.act(state, eps=eps)
            #a = env.step(action)
            next_state, reward, done, info = env.step(action)
            if self.render:
                env.render()

            if not (self.optimal):
                loss = self.step(state, action, reward, next_state, done)
                loss_list.append(loss)

            state = next_state.copy()
            score += reward
            if done:
                #env.plot_state()

                if env.score > env.high_score:
                    print(" ")
                    print("New high socre: ", env.score)
                break
            _+=1

        return score,_, loss_list,eps

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                return loss
        return np.nan



    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            Q_targets = (rewards + (1 - dones) * self.gamma * torch.max(self.qnetwork_target(next_states),
                                                                        dim=1).values)[:, None]
        # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)) #.permute((0,3,1,2))
        Q_expected = self.qnetwork_local(states)[torch.arange(self.batch_size), actions].unsqueeze(1)

        # Compute loss
        #loss = F.mse_loss(Q_expected, Q_targets)
        loss = F.smooth_l1_loss(Q_expected,Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return loss.item()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state) #no permutation .permute((0,3,1,2))
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.qnetwork_local.action_size))

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
