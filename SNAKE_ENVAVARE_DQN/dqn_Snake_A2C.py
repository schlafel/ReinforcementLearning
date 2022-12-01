import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers
import snake_gymDirected
import matplotlib.pyplot as plt
import tqdm
import tensorflow_probability as tfp
import pandas as pd

#https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=num_states)
        self.conv0 = tf.keras.layers.Conv2D(64, (3, 3),
                                            padding='same',
                                            activation='relu')

        self.conv1 = tf.keras.layers.Conv2D(2, (1, 1),
                                            padding='same',
                                            activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i,  kernel_initializer='RandomNormal'))



        self.output_layer_value = tf.keras.layers.Dense(
            1, kernel_initializer='RandomNormal')
        self.output_layer_action = tf.keras.layers.Dense(
            num_actions, activation='softmax', kernel_initializer='RandomNormal')

    # @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        z = self.conv0(z)
        z = self.conv1(z)
        z = self.flatten(z)
        for layer in self.hidden_layers:
            z = layer(z)

        output_policy = self.output_layer_action(z)


        output_value = self.output_layer_value(z)

        return output_policy, output_value



class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(inputs)

    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss

    def train(self, TargetNet):
        #https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97
        if len(self.experience['s']) < self.min_experiences:
            return 0,0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i][:,:] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i][:,:] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        # state = np.array([state])
        # next_state = np.array([next_state])
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # p = self.actor(states, training=True)
            # v =  self.critic(states,training=True)
            p,v = TargetNet.model(states, training=True)

            pn,vn = TargetNet.model(states_next, training=True)
            # vn = self.critic(next_state, training=True)
            td = rewards + self.gamma*vn*(1-(1*dones)) - v
            a_loss = self.actor_loss(p, actions, td)
            c_loss = td**2
        grads1 = tape1.gradient(a_loss, TargetNet.model.trainable_variables)
        grads2 = tape2.gradient(c_loss, TargetNet.model.trainable_variables)
        TargetNet.optimizer.apply_gradients(zip(grads1, TargetNet.model.trainable_variables))
        TargetNet.optimizer.apply_gradients(zip(grads2, TargetNet.model.trainable_variables))
        return a_loss, c_loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(env, TrainNet, TargetNet, epsilon, copy_step,n,show_ep = True,show_plotAfter = 50000):
    rewards = 0
    iteration = 0
    done = False
    observations = env.reset()
    losses = list()

    if show_ep &( (n %show_plotAfter) == 0):
        env.render()
        show_ep2 = True
    else:
        show_ep2 = False

    while not done:
        action = TrainNet.get_action(observations[None,:,:,:], epsilon)
        prev_observations = observations[:,:,:]
        observations, reward, done, _ = env.step(action)

        if show_ep2:
            env.render()

        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp = {'s': prev_observations,
               'a': action,
               'r': reward,
               's2': observations[:,:,:],
               'done': done}

        TrainNet.add_experience(exp)
        a_loss,c_loss = TrainNet.train(TargetNet)
        if isinstance(a_loss, int):
            losses.append(a_loss)
        else:
            losses.append(tf.reduce_mean(a_loss).numpy())
        iteration += 1
        if iteration % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses),iteration

def make_video(env, TrainNet):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = TrainNet.get_action(observation, 0)
        observation, reward, done, _, _= env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main(env,env_name = "snake"):

    gamma = 0.99
    copy_step = 25
    num_states = env.observation_space.shape
    num_actions = env.action_space.n
    hidden_units = [64]
    max_experiences = 100000
    min_experiences = 100
    batch_size = 32
    lr = 1e-3
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn_{}/'.format(env_name) + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TrainNet.model(env.reset()[None, :, :, :])
    TrainNet.model.compile()
    TrainNet.model.summary()
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet.model(env.reset()[None, :, :, :])
    TargetNet.model.compile()
    TargetNet.model.summary()
    N = 1000
    total_rewards = np.empty(N)
    total_iter = np.zeros(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    show_plotAfter = 20
    for n in tqdm.tqdm(range(N)):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses,iters = play_game(env, TrainNet, TargetNet, epsilon, copy_step,n,
                                               show_ep = True,
                                               show_plotAfter=show_plotAfter)
        total_rewards[n] = total_reward
        total_iter[n] = iters
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss', losses, step=n)
            tf.summary.scalar('average iterations', iters, step=n)
            tf.summary.scalar('Score', env.score, step=n)
            tf.summary.scalar('High-Score', env.high_score, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward,
                  "eps:", epsilon,
                  "avg reward (last 100):", avg_rewards,
                  "Average iterations (last 100):", np.mean(total_iter[n-100:n]),
                  "episode loss: ", losses)
        if n % 1000 == 0:
            TrainNet.model.save(os.path.join(log_dir,"DQN_Snake_{:d}".format(n)))
            

    # Include the epoch in the file name (uses `str.format`)

    TrainNet.model.save(os.path.join(log_dir,"DQN_Snake_{:d}".format(n)))
    print("avg reward for last 100 episodes:", avg_rewards)
    make_video(env, TrainNet)
    env.close()


if __name__ == '__main__':

    for i in range(3):
        env = gym.make("SnakeDir-v0", env_config={"gs":(12,12),
                                               "BLOCK_SIZE": 20},
                       )
        main(env)