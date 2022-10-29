import tensorflow as tf
import random
import numpy as np
from collections import deque
from game import SnakeGameAI,Direction,Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #parameter for randomness
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY) #automatically remove popleft()

        #model, trainer



    def get_state(self,game):
        ## 11 Values
        #[danger_straight, danger_right,danger_left,danger_right
        head = game.snake[0]
        bs = game.BLOCK_SIZE
        point_l = Point(head.x - bs,head.y)
        point_r = Point(head.x + bs,head.y)
        point_u = Point(head.x,head.y-bs)
        point_d = Point(head.x,head.y+bs)


        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            #Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            #Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),


            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,


            #Food location
            game.food.x < game.head.x, #Food to the left
            game.food.x > game.head.x, #Food to the right
            game.food.y < game.head.y,
            game.food.y > game.head.y

        ]

        return np.array(state,dtype = int)

        pass

    def remember(self,state,action,reward,next_state,game_over):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self,state,action,reward,next_state,game_over):
        pass

    def get_action(self,state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        #get old state
        state_old = Agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new wtate
        reward, game_over,score = game.play_step(final_move)
        state_new = agent.get_state(game)


        #train short memory (only 1 step)
        agent.train_short_memory(state_old,final_move,reward,state_new,game_over)

        #remember
        agent.remember(state_old,final_move,reward,state_new,game_over)

        if game_over:
            #train long (experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()
            print("Game",agent.n_games,"Score",score,"Record",record)

    pass




if __name__ == '__main__':
    print("yes")
    train()
