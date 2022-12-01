from model import A2CAgent
import gym
import snake_gymDirected
import numpy as np


class SnakeLearningEnv():
    def __init__(self,gs = (12,12),batch_size = 128):
        self.input_shape = gs
        self.env = gym.make("SnakeDir-v0", env_config={"gs": gs,
                                                  "BLOCK_SIZE": 20},)

        self.batch_size = batch_size
        self.a2_ag = A2CAgent(number_actions = self.env.action_space.n,
                              input_shape = gs)
        self.initialize_models()

        self.memory = ReplayBuffer(size = 10000,obs_dim=self.env.reset().shape)


    def initialize_models(self):
        state0 = self.env.reset()
        #get summaries
        self.a2_ag.actor.predict(state0[None, :, :, :])
        self.a2_ag.actor.summary()

        self.a2_ag.critic.predict(state0[None, :, :, :])
        self.a2_ag.critic.summary()

        print(30*"*", "Actor ", 30*"*")
        self.a2_ag.actor.summary()

        print(30*"*", "Critic ", 30*"*")
        self.a2_ag.critic.summary()



    def train(self,steps = 3000,render = False):

        for s in range(steps):

            done = False
            state = self.env.reset()
            total_reward = 0
            all_aloss = []
            all_closs = []

            while not done:
                if render:
                    self.env.render()
                action = self.a2_ag.act(state)
                # print(action)
                next_state, reward, done, _ = self.env.step(action)
                # self.env.plot_state()
                # aloss, closs = self.a2_ag.learn(state[None,:,:,:], action, reward, next_state[None,:,:,:], done*1)
                # all_aloss.append(aloss)
                # all_closs.append(closs)
                state = next_state
                total_reward += reward

                #add to memory
                self.memory.add(state,action,reward,next_state,done)

                if done:
                    # print("total step for this episord are {}".format(t))
                    print("total reward after {} steps is {}".format(s, total_reward))

            #now learn from memory
            if len(self.memory) >  self.batch_size:
                print("Training....")
                experiences = self.memory.sample()
                self.a2_ag.learn(*experiences)

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim, size: int, batch_size: int = 32):

        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0


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
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return (self.obs_buf[idxs],
               self.acts_buf[idxs],
                self.rews_buf[idxs],
                self.next_obs_buf[idxs],
                self.done_buf[idxs])

    def __len__(self):
        return self.size
def main():
    learner = SnakeLearningEnv(batch_size=1024)
    learner.train(steps = 3000,render = True,)



if __name__ == '__main__':

    # https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97
    # https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-2of-2-b8ceb7e059db

    main()
