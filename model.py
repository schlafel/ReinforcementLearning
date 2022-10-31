import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
import os
import numpy as np

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


class Linear_QNet(tf.keras.Model):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()

        self.create(input_size,hidden_size,output_size)
        self.call(Input(shape=(11,)))


    def create(self,input_size,hidden_size,output_size):
        self.dense1 = Dense(input_size)
        self.dense2 = Dense(hidden_size)
        self.output_layer = Dense(output_size,activation = "linear")

        self.compile(tf.keras.optimizers.Adam(learning_rate=1e-3),
                     )

    def call(self,x):
        #x = Input(x,)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

    def save(self,model_path = "models"):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save(os.path.join(model_path,"model.h5"))

    class ModelSaver(tf.keras.callbacks.Callback):
        def __init__(self):
            super.__init__()


class QTrainer:
    def __init__(self, model, lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = tf.keras.optimizers.Adam(learning_rate = self.lr)
        self.criterion = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(self,state,action,reward,next_state,game_over):
        state = tf.convert_to_tensor(state,dtype = tf.float32)
        next_state = tf.convert_to_tensor(next_state,dtype = tf.float32)
        action = tf.convert_to_tensor(action,dtype = tf.int)
        reward = tf.convert_to_tensor(reward,dtype = tf.int)

        if len(state.shape) == 1:
            pass
            #ev. reshape.... for state,next_state,action,reward,game_over

        #implement Bellmann-equation
        #get predicted Q-Values with the current state
        pred = self.model.predict(state) #this is an action....

        #apply r+ y * max(next_predQ)
        target = pred.copy()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:

                Q_new = reward[idx] + self.gamma * tf.reducemax(self.model.predict(next_state))

            target[idx][np.argmax(action)] = Q_new

        #do the gradients in tensorflow

        with tf.GradientTape() as gradTape:

            loss = self.criterion(target,pred) #calculate the loss (mse)

        gradients_of_disc2 = gradTape.gradient(loss, self.model.trainable_variables)

        # parameters optimization for discriminator for fake labels
        self.optim.apply_gradients(zip(gradients_of_disc2,
                                                    self.model.trainable_variables))


if __name__ == '__main__':
    #testing
    mod = Linear_QNet(11,7,3)

    mod.save()
    mod.summary()


