import pygame
import random
from enum import Enum
from collections import namedtuple
import time
import numpy as np
Point = namedtuple('Point','x, y')
BLOCK_SIZE=200
snake_speed = 15

pygame.init()
# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4



class SnakeGameAI:
    font = pygame.font.SysFont('arial', 25)
    def __init__(self,
                 snake_speed = 15,
                 w = 640,h=480,
                 BLOCK_SIZE = 20):
        self.w = w
        self.h = h
        self.BLOCK_SIZE = BLOCK_SIZE

        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption("Snake")

        self.clock = pygame.time.Clock()
        self.reset()

        self.n_rows = h // BLOCK_SIZE
        self.n_cols = w // BLOCK_SIZE






    def reset(self):
        #initial direction
        self.direction = Direction.RIGHT


        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*self.BLOCK_SIZE), self.head.y),]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration =  0






    def _place_food(self):
        x = random.randint(0,(self.w-self.BLOCK_SIZE)//self.BLOCK_SIZE)*self.BLOCK_SIZE
        y = random.randint(0,(self.h-self.BLOCK_SIZE)//self.BLOCK_SIZE)*self.BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self,action):
        self.frame_iteration+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.key == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:
            #         self.direction = Direction.DOWN

        self._move(action)
        self.snake.insert(0,self.head)


        game_over = False
        reward = 0
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        #Move head
        if self.head == self.food:
            reward = 10
            #add score
            self.score +=1
            self._place_food()
        else:
            self.snake.pop()


        self._update_ui()
        self.clock.tick(snake_speed)
        self.get_observation()

        return reward, game_over, self.score,self.state

    def _move(self,action):

        clock_wise = [Direction.RIGHT,
                      Direction.DOWN,
                      Direction.LEFT,
                      Direction.UP]

        idx = clock_wise.index(self.direction)

        if np.array_equal(action,[1,0,0]):
            #keep direction
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx+1)%4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x+= self.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.BLOCK_SIZE

        self.head = Point(x,y)

    def is_collision(self,pt = None):
        if pt is None:
            pt = self.head
        if pt.x > (self.w - self.BLOCK_SIZE) or pt.x < 0 or pt.y > (self.h - self.BLOCK_SIZE) or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(black)

        for pt in self.snake:
            pygame.draw.rect(self.display,green,pygame.Rect(pt.x,pt.y,self.BLOCK_SIZE,self.BLOCK_SIZE))
        pygame.draw.rect(self.display,white,pygame.Rect(self.food.x,self.food.y,self.BLOCK_SIZE,self.BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, white)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def set_borders(self):
        # border
        self.state[0, :,0] = 1
        self.state[-1, :,0] = 1
        self.state[ :, 0,0] = 1
        self.state[:, -1,0] = 1


    def get_observation(self):
        #reset state
        self.state = np.zeros((self.n_rows + 2, self.n_cols + 2,2))
        self.set_borders()
        #update the state (array)
        #put snake in array (0th channel)
        for pt in self.snake:
            self.state[int(pt.y // self.n_rows + 1), int(pt.x // self.n_cols + 1), 1] = 1.0

        #place Food
        self.state[int(self.food.y // self.n_rows + 1), int(self.food.x // self.n_cols + 1), 0] = -1.0
        return self.state





if __name__ == '__main__':
    print("Starting pygame")
    game = SnakeGameAI()

    while True:
        game_over, score = game.play_step()
        if game_over == True:
            break


    print("Final score: ",score)

    pygame.quit()