import tensorflow as tf
import random
import numpy as np
from collections import deque
from game import SnakeGameAI,Direction,Point
import tensorflow as tf
from model import Linear_QNet,QTrainer

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
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(model = self.model,
                                lr = LR,
                                gamma = self.gamma)



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



    def remember(self,state,action,reward,next_state,game_over):
        self.memory.append((state,action,reward,next_state,game_over)) #poplef if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) #get BATCH_SIZE --> list of tuples
        else:
            mini_sample = self.memory

        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self,state,action,reward,next_state,game_over):
        self.trainer.train_step(state,action,reward,next_state,game_over)

    def get_action(self,state):
        #random moves: tradeoff between exploration / extploitation

        self.epsilon = 80 - self.n_games
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



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        #get old state
        state_old = agent.get_state(game=game)

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






if __name__ == '__main__':
    print("yes")
    train()
