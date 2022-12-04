import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
import os
import numpy as np

# Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')


class Linear_QNet(tf.keras.Model):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()

        self.create(input_size,hidden_size,output_size)
        self.call(Input(shape=(11,)))


    def create(self,input_size,hidden_size,output_size):
        self.dense1 = Dense(input_size,activation = "linear")
        self.dense2 = Dense(hidden_size,activation = "ReLU")
        self.output_layer = Dense(output_size,activation = "linear")

        # self.compile(tf.keras.optimizers.Adam(learning_rate=1e-3),
        #              )

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

    # @tf.function()
    def train_step(self,state,action,reward,next_state,game_over):
        state = tf.convert_to_tensor(state,dtype = tf.float32)
        next_state = tf.convert_to_tensor(next_state,dtype = tf.float32)
        action = tf.convert_to_tensor(action,dtype = tf.int32)
        reward = tf.convert_to_tensor(reward,dtype = tf.float32)

        if len(state.shape) == 1:
            state = tf.expand_dims(state,axis = 0)
            next_state = tf.expand_dims(next_state,axis = 0)
            action = tf.expand_dims(action,axis = 0)
            reward = tf.expand_dims(reward,axis = 0)
            game_over = (game_over,)
            #ev. reshape.... for state,next_state,action,reward,game_over


        #implement Bellmann-equation
        #get predicted Q-Values with the current state
        # pred = tf.make_ndarray(self.model(state,training = False)) #this is an action....
        pred = self.model(state,training = False)
        #apply r+ y * max(next_predQ)
        target = tf.identity(pred).numpy()
        # target = pred.copy()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:

                Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model(next_state,training = False))

            target[idx][np.argmax(action)] = Q_new

        #do the gradients in tensorflow

        with tf.GradientTape() as gradTape:

            loss = self.criterion(tf.convert_to_tensor(target),
                                  self.model(state,training = True)) #calculate the loss (mse) #importatnt to use self.model(state)

        gradients_of_disc2 = gradTape.gradient(loss, self.model.trainable_variables)

        # parameters optimization for discriminator for fake labels
        self.optim.apply_gradients(zip(gradients_of_disc2,
                                                    self.model.trainable_variables))


if __name__ == '__main__':
    #testing
    mod = Linear_QNet(11,256,3)

    mod.predict(np.random.random((1, 11)))
    mod.summary()
    mod.save()
    mod.summary()


