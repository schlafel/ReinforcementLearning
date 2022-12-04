import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
import datetime
import matplotlib.pyplot as plt
import snake_gymDirected
import snake_gym
import os
import tqdm

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(64, (3, 3),
                                            padding='same',
                                            activation='relu',)

        self.conv1 = tf.keras.layers.Conv2D(1, (1, 1),
                                            padding='same',
                                            activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        # self.d2 = tf.keras.layers.Dense(32,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        # x = self.conv0(input_data)
        # x = self.conv1(x)
        x = self.flatten(input_data)
        x = self.d1(x)
        # x = self.d2(x)
        v = self.v(x)
        return v


class Actor(tf.keras.Model):
    def __init__(self,n_actions = 4):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(64, (3, 3),
                                            padding='same',
                                            activation='relu',)

        self.conv1 = tf.keras.layers.Conv2D(1, (1, 1),
                                            padding='same',
                                            activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        # self.d2 = tf.keras.layers.Dense(32,activation='relu')
        self.a = tf.keras.layers.Dense(n_actions, activation='softmax')

    def call(self, input_data):
        # x = self.conv0(input_data)
        # x = self.conv1(x)
        x = self.flatten(input_data)
        x = self.d1(x)
        # x = self.d2(x)
        a = self.a(x)
        return a


class Agent():
    def __init__(self, gamma=0.99,n_actions = 4):

        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.actor = Actor(n_actions=n_actions)
        self.critic = Critic()



    def act(self, state):
        prob = self.actor(state[None,:,:,:])
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])


    def actor_loss(self, probs, actions, td):

        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        # print(probability)
        # print(log_probability)

        p_loss = []
        e_loss = []
        td = td.numpy()
        # print(td)
        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        # print(p_loss)
        # print(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        # print(loss)
        return loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            # print(discnt_rewards)
            # print(v)
            # print(td.numpy())
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss


def preprocess1(states, actions, rewards, gamma):
    discnt_rewards = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
        sum_reward = r + gamma * sum_reward
        discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)

    return states, actions, discnt_rewards

if __name__ == '__main__':

    tf.random.set_seed(336699)
    env_name = "LunarLander-v2"
    env_name = "SnakeDir-v0"
    # env_name = "Snake-v0"
    render = False

    # env = gym.make("CartPole-v0")
    env = gym.make(env_name, env_config={"gs": (8,8),
                                                  "BLOCK_SIZE": 20},)
    # env = gym.make(env_name,
    #                # env_config={"gs": (12,12),
    #                #                                "BLOCK_SIZE": 20},
    #                )

    agentoo7 = Agent(n_actions=env.action_space.n,
                     )

    state = env.reset()
    agentoo7.actor(state[None,:,:,:])
    agentoo7.critic(state[None,:,:,:])

    agentoo7.actor.summary()
    agentoo7.critic.summary()



    steps = 10000

    low = env.observation_space.low
    high = env.observation_space.high

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = 'logs/a2c_{}/'.format(env_name) + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    ep_reward = []
    total_avgr = []
    for s in tqdm.tqdm(range(steps)):

        done = False
        state = env.reset()
        total_reward = 0
        all_aloss = []
        all_closs = []
        rewards = []
        states = []
        actions = []

        while not done:

            action = agentoo7.act(state)
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()
            rewards.append(reward)
            states.append(state)
            # actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)

                states, actions, discnt_rewards = preprocess1(states, actions, rewards, 1)
                al, cl = agentoo7.learn(states, actions, discnt_rewards)
                if s % 100 == 0:
                    print(" ")
                    print("total reward after {} steps is {} and avg reward is {}".format(s, total_reward, avg_reward))

                    print(f"al{al}")
                    print(f"cl{cl}")

                    with summary_writer.as_default():
                        tf.summary.scalar('episode reward', total_reward, step=s)
                        tf.summary.scalar('running avg reward(100)', avg_reward, step=s)
                        tf.summary.scalar('actor loss', al, step=s)
                        tf.summary.scalar('critics loss', cl, step=s)
                        tf.summary.scalar('Score', total_reward, step=s)

        if s%1000 == 0:
            agentoo7.actor.save_weights(os.path.join(log_dir,env_name +"_Actor_" +str(s)))
            agentoo7.critic.save_weights(os.path.join(log_dir,env_name +"_Critic_" +str(s)))
    ep = [i for i in range(steps)]
    plt.plot(ep, total_avgr, 'b')
    plt.title("avg reward Vs episods")
    plt.xlabel("episods")
    plt.ylabel("average reward per 100 episods")
    plt.grid(True)
    plt.show()