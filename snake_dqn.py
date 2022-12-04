import tensorflow as tf
import random
import numpy as np
from collections import deque
import gym
import snake_gymDirected
import tensorflow as tf


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


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

        if len(state.shape) == 3:
            state = tf.expand_dims(state,axis = 0)
            next_state = tf.expand_dims(next_state,axis = 0)
            action = tf.expand_dims([action],axis = 1)
            reward = tf.expand_dims(reward,axis = 0)
            game_over = (game_over,)
            #ev. reshape.... for state,next_state,action,reward,game_over
        else:
            action = tf.expand_dims(action, axis=0)


        #implement Bellmann-equation
        #get predicted Q-Values with the current state
        # pred = tf.make_ndarray(self.model(state,training = False)) #this is an action....
        pred = self.model(state,training = False)
        #apply r+ y * max(next_predQ)
        target = tf.identity(pred).numpy().copy()
        # target = pred.copy()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model(next_state,training = False))

            #target[idx][np.argmax(action)] = Q_new
            target[idx][action[0,idx]] = Q_new

        #do the gradients in tensorflow

        with tf.GradientTape() as gradTape:
            loss = self.criterion(tf.convert_to_tensor(target),
                                  self.model(state,training = True)) #calculate the loss (mse) #importatnt to use self.model(state)

        gradients_of_disc2 = gradTape.gradient(loss, self.model.trainable_variables)

        # parameters optimization for discriminator for fake labels
        self.optim.apply_gradients(zip(gradients_of_disc2,
                                                    self.model.trainable_variables))



class SnakeEnvModel(tf.keras.Model):
    def __init__(self,out_size = 3):
        super().__init__()

        self.conv1 =   tf.keras.layers.Conv2D(64, (3, 3),
                                            padding='same',
                                            activation='relu')

        self.conv2 = tf.keras.layers.Conv2D(32, (1, 1),
                                            padding='same',
                                            activation='relu')

        self.flatten1 = tf.keras.layers.Flatten()
        #self.dense1 = tf.keras.layers.Dense(32)
        self.output_layer = tf.keras.layers.Dense(out_size)

    def call(self,x):
        #x = Input(x,)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten1(x)
        #x = self.dense1(x)
        return self.output_layer(x)

class Agent:
    def __init__(self,input_shape = None, n_actions = 3):
        self.n_games = 0
        self.epsilon = 0 #parameter for randomness
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) #automatically remove popleft()
        self.n_actions = n_actions
        #model, trainer
        self.model = SnakeEnvModel(out_size=self.n_actions)

        self.model.build(input_shape = (None,input_shape[0],
                                        input_shape[1],
                                        input_shape[2]))
        self.model.call(tf.keras.layers.Input(input_shape))
        self.model.summary()
        self.trainer = QTrainer(model = self.model,
                                lr = LR,
                                gamma = self.gamma)

    def remember(self,state,action,reward,next_state,game_over):
        self.memory.append((state,action,reward,next_state,game_over)) #poplef if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) #get BATCH_SIZE --> list of tuples
        else:
            mini_sample = self.memory

        states,actions,rewards,next_states,game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    def train_short_memory(self,state,action,reward,next_state,game_over):
        self.trainer.train_step(state,action,reward,next_state,game_over)

    def get_action(self,state):
        #random moves: tradeoff between exploration / extploitation

        self.epsilon = 80 - self.n_games
        final_move = 0
        if random.randint(0,200) < self.epsilon:
            final_move = random.randint(0,self.n_actions-1)
        else:
            #is that really a tensor?
            state0 = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
            prediction = self.model.predict(state0)

            final_move = np.argmax(prediction)

        return final_move



def train(env,render = True,n_epochs=1000):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    state_old = env.reset()


    agent = Agent(input_shape = state_old.shape,
                  n_actions = env.action_space.n)

    if render:
        env.render()
    for epoch in range(n_epochs):
        game_over = False
        env.reset()
        while not game_over:
            #get old state
            state_old = env.get_observation()

            #get move
            final_move = agent.get_action(state_old)

            #perform move and get new wtate
            next_state, reward, game_over,_ = env.step(final_move)
            #state_new = agent.get_state(game)
            if render:
                env.render()

            #train short memory (only 1 step)
            agent.train_short_memory(state_old,final_move,reward,next_state,game_over)

            #remember
            agent.remember(state_old,final_move,reward,next_state,game_over)

            if game_over:
                #train long (experience replay)

                agent.n_games += 1
                agent.train_long_memory()

                if env.score > record:
                    record = env.score
                    #agent.model.save()
                print("Epoch:",epoch,"Game",agent.n_games,"Score",env.score,"Record",record)






if __name__ == '__main__':
    print("yes")
    env = gym.make("SnakeDir-v0", env_config={"gs": (12, 12),
                                              "BLOCK_SIZE": 20,
                                              "snake_length":0},
                   )


    N_EPOCHS = 1000
    train(env = env,
          render = True,
          n_epochs=N_EPOCHS)
