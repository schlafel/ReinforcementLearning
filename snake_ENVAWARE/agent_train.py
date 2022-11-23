from game import SnakeGameAI,Direction,Point
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from collections import deque
import random
import tensorflow as tf
matplotlib.use("TkAgg")

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #parameter for randomness
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) #automatically remove popleft()


        #model, trainer
        # self.model = Linear_QNet(11,256,3)
        # self.trainer = QTrainer(model = self.model,
        #                         lr = LR,
        #                         gamma = self.gamma)
        #
    def get_action(self,state):
        #random moves: tradeoff between exploration / extploitation

        self.epsilon = 200 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            #is that really a tensor?
            state0 = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
            prediction = self.model.predict(state0)

            move = np.argmax(prediction)
            final_move[move] = 1
        return final_move





if __name__ == '__main__':
    game = SnakeGameAI()
    agent = Agent()

    while True:
        state_old = game.get_observation()
        #get Move
        final_move = agent.get_action(state_old)



        #perform move and get new wtate
        reward, game_over, score,state = game.play_step(final_move)
        state_new = game.get_observation()
        print("played a step! :-) ")