import pygame
import random
from enum import Enum
from collections import namedtuple
import time

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



class SnakeGame:
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

        #initial direction
        self.direction = Direction.RIGHT


        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*self.BLOCK_SIZE), self.head.y),]

        self.score = 0
        self.food = None
        self._place_food()


    def _place_food(self):
        x = random.randint(0,(self.w-self.BLOCK_SIZE)//self.BLOCK_SIZE)*self.BLOCK_SIZE
        y = random.randint(0,(self.h-self.BLOCK_SIZE)//self.BLOCK_SIZE)*self.BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        self._move(self.direction)
        self.snake.insert(0,self.head)


        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        #Move head
        if self.head == self.food:
            #add score
            self.score +=1
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(snake_speed)

        return game_over, self.score

    def _move(self,direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x+= self.BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif direction == Direction.UP:
            y -= self.BLOCK_SIZE

        self.head = Point(x,y)

    def _is_collision(self):
        if self.head.x > (self.w - self.BLOCK_SIZE) or self.head.x < 0 or self.head.y > (self.h - self.BLOCK_SIZE) or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(black)
        pygame.draw.rect(self.display, red, pygame.Rect(self.snake[0].x, self.snake[0].y, self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(self.display, black,
                         pygame.Rect(self.snake[0].x+5, self.snake[0].y+5, self.BLOCK_SIZE - 10, self.BLOCK_SIZE - 10))

        for pt in self.snake[1:]:
            pygame.draw.rect(self.display,green,
                             pygame.Rect(pt.x+2,pt.y+2,self.BLOCK_SIZE-4,self.BLOCK_SIZE-4))
            pygame.draw.rect(self.display, black, pygame.Rect(pt.x + 5, pt.y + 5, self.BLOCK_SIZE - 10, self.BLOCK_SIZE - 10))
            pygame.draw.circle(self.display,red,(pt.x + 5 ,
                                                 pt.y + 5), 2,2)
        pygame.draw.rect(self.display,white,pygame.Rect(self.food.x,self.food.y,self.BLOCK_SIZE,self.BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, white)
        self.display.blit(text, [0, 0])
        pygame.display.flip()



if __name__ == '__main__':
    print("Starting pygame")
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()
        if game_over == True:
            break


    print("Final score: ",score)

    pygame.quit()