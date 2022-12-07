import random

import numpy as np

from snake_gym.envs import SnakeEnvV0
from snake_gym.envs.snake_env import Point
import gym


class SnakeEnvV0PyTorch(SnakeEnvV0):
    #Agent with Borders and config for pytroch
    def __init__(self,action_map=None,
                 env_config=None,
                 render_mode=None):
        super().__init__(action_map=action_map,
                 env_config=env_config,
                 render_mode=None)
        self.action_map = {
            0: 'right',
            1: 'down',
            2: 'left',
            3: 'up',
        }

        self.action_space = gym.spaces.Discrete(len(self.action_map.keys()))

    def get_observation(self):
        # reset state
        if self.state is None:
            state_old = np.zeros((1, self.n_rows, self.n_cols),
                     dtype=float)
            state_old = self.set_borders(state_old)
        else:
            state_old = self.state[[0],:,:]

        self.state = np.zeros((1,self.n_rows, self.n_cols),
                              dtype=float)
        self.calc_distance_food()
        self.state = self.set_borders(self.state)
        # update the state (array)
        # put snake in array (0th channel)
        for _i,pt in enumerate(reversed(self.snake)):
            # self.state[int(pt.y // self.n_rows), int(pt.x // self.n_cols), 0] = .5
            self.state[0, int(pt.y /self.BLOCK_SIZE),
                       int(pt.x / self.BLOCK_SIZE)] = 1
        #set Head to 0
        self.state[0, int(pt.y / self.BLOCK_SIZE), int(pt.x / self.BLOCK_SIZE)] = 3

        # place Food
        self.state[0, int(self.food.y / self.BLOCK_SIZE),
                   int(self.food.x / self.BLOCK_SIZE)] = -20

        self.info = dict({"score":self.score,
                          "food_distance":self.food_distance})
        return np.concatenate([self.state,state_old])

    def set_borders(self,array):
        array[0,:,0] = 1
        array[0, 0,:] = 1
        array[0,:,self.n_cols-1] = 1
        array[0,self.n_rows-1,:] = 1
        return array


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # if pt.x > (self.w - self.BLOCK_SIZE) or pt.x < 0 or pt.y > (self.h - self.BLOCK_SIZE) or pt.y < 0:
        if pt.x > (self.w - 2*self.BLOCK_SIZE) or pt.x < self.BLOCK_SIZE or pt.y > (self.h - 2*self.BLOCK_SIZE) or pt.y < self.BLOCK_SIZE:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _place_food(self):
        x = random.randint(1, (self.w - 2*self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
        y = random.randint(1, (self.h - 2*self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()



    def step(self, action:int):
        # perform one step in the game logic


        # print(action)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        old_food_distance = self.food_distance
        self._move(action)
        self.snake.insert(0, self.head)

        food_distance = self.calc_distance_food()
        # print(food_distance - old_food_distance)

        game_over = False
        self.game_over = game_over
        reward = -.1 if (food_distance - old_food_distance) < 0 else .1

        #check if collision
        if self.is_collision() or self.frame_iteration > 25 * len(self.snake):
            game_over = True
            self.game_over = game_over
            reward = -10
            self.get_observation()

            return self.state, reward, game_over, False,self.info

        # Move head
        if self.head == self.food:
            reward = 10
            # add score
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
            self._place_food()
        else:
            self.snake.pop()

        self.state = self.get_observation()
        self.info = dict({"score":self.score})

        #perform render step
        if self.screen is not None:
            self.renderer.render_step()

        return self.state, reward, game_over, False, self.info,
