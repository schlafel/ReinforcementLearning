import tensorflow as tf
from tensorflow import keras
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #Entry Layer
        self.conv0 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')


        #repeated layer
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')


        #value head layer
        self.ValueHead_conv = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='relu')
        self.ValueHead_flatten = tf.keras.layers.Flatten()
        self.ValueHead_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.ValueHead_dense2 = tf.keras.layers.Dense(1)

        #PolicyHeadLayer
        self.PolHead_conv = tf.keras.layers.Conv2D(2, (1, 1), padding='same', activation='relu')
        self.PolHead_flatten = tf.keras.layers.Flatten()
        self.PolHead_dense = tf.keras.layers.Dense(4)

        self.act = tf.keras.layers.Activation('softmax')



    def call(self,input_tensor, training = False):
        x = self.conv0(input_tensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #ValueHead
        value_output = self.ValueHead_conv(x)
        value_output = self.ValueHead_flatten(value_output)
        value_output = self.ValueHead_dense1(value_output)
        value_output = self.ValueHead_dense2(value_output)


        #self.critic_head = ValueHeadLayer()

        #criticHEad
        critic_output = self.PolHead_conv(x)
        critic_output = self.PolHead_flatten(critic_output)
        critic_output = self.PolHead_dense(critic_output)

        return critic_output,self.act(critic_output),value_output

    def action_value(self, obs):
        logits,_, value = self.predict(obs)

        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        return action, np.squeeze(value, axis=-1)


    def top_action(self,obs):
        logits,_, _ = self.predict(obs)
        action = tf.argmax(logits, 1)
        return np.squeeze(action, axis=-1)



class A2CAgent:
    def __init__(self, model,lr = 0.0007):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
        )

    def initialize_model(self, env):
        a = env.get_observation()/255.0

        self.model.action_value(a[None, :])
        self.model.top_action(a[None, :])

    @staticmethod
    @tf.function
    def calc_entropy(logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - self.params['entropy'] * entropy_loss

    def _value_loss(self, returns, value):
        return self.params['value'] * tf.keras.losses.mean_squared_error(returns, value)



    @tf.function
    def train(self, states, rewards, values, actions):
        advs = rewards - values
        with tf.GradientTape() as tape:
            logits,_,vals = self.model(states)
            neglogpac = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(actions, 4))
            _policy_loss = neglogpac*tf.squeeze(advs)
            policy_loss = tf.reduce_mean(_policy_loss)

            #vpred = self.model.vcall(states)
            value_loss = tf.reduce_mean(tf.square(vals-rewards))

            entropy = tf.reduce_mean(self.calc_entropy((logits)))

            loss = policy_loss + value_loss * 0.5 - 0.1 * entropy

        var_list = tape.watched_variables()
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.model.optimizer.apply_gradients(zip(grads, var_list))

        return loss, policy_loss, value_loss,






