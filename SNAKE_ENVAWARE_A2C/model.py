import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp




class Actor(tf.keras.Model):
    def __init__(self,number_actions = 3,input_shape = None):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(64, (3, 3),
                                            padding='same',
                                            activation='relu',
                                            input_shape = input_shape)

        self.conv1 = tf.keras.layers.Conv2D(2, (1, 1),
                                            padding='same',
                                            activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        #self.d1 = tf.keras.layers.Dense(2048, activation='relu')
        #self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.a = tf.keras.layers.Dense(number_actions, activation='softmax')

    def call(self, input_data):
        x = self.conv0(input_data)
        x = self.conv1(x)
        x = self.flatten(x)
        a = self.a(x)
        return a

class Critic(tf.keras.Model):
    def __init__(self,input_shape = None):
        super().__init__(self)

        self.conv0 = tf.keras.layers.Conv2D(64, (3, 3),
                                            padding='same',
                                            activation='relu',input_shape = input_shape)

        self.conv1 = tf.keras.layers.Conv2D(1, (1, 1),
                                            padding='same',
                                            activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        #self.d1 = tf.keras.layers.Dense(2048, activation='relu')
        #self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.conv0(input_data)
        x = self.conv1(x)
        x = self.flatten(x)
        v = self.v(x)
        return v



class A2CAgent():
    def __init__(self,
                 gamma = 0.99,
                 lr = 1e-4,
                 number_actions = 1,
                 input_shape = None):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.actor = Actor(number_actions = number_actions,input_shape=input_shape)
        self.critic = Critic(input_shape=input_shape)

    def act(self, state):
        prob = self.actor(np.array([state]))
        # print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])


    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss


    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(state, training=True)
            v =  self.critic(state,training=True)
            vn = self.critic(next_state, training=True)
            td = reward + self.gamma*vn*(1-int(done)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td**2
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

