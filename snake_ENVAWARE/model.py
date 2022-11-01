import tensorflow as tf
import nump as np

class Model(tf.keras.Model):
    def __init__(self):
        super().__init()
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
        self.PolHead_dense = tf.keras.layers.Dense(3)



    def call(self,input_tensor, training = False):
        x = self.conv0(input_tensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #ValueHead
        actor_output = self.ValueHead_conv(x)
        actor_output = self.ValueHead_flatten(actor_output)
        actor_output = self.ValueHead_dense1(actor_output)
        actor_output = self.ValueHead_dense2(actor_output)


        #criticHEad
        critic_output = self.PolHead_conv(x)
        critic_output = self.PolHead_flatten(critic_output)
        critic_output = self.PolHead_dense(critic_output)

        return actor_output,critic_output

    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.sample_action_from_logits.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)










