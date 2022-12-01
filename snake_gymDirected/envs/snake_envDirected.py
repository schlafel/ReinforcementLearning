import pygame
import numpy as np
import gym
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer
from enum import Enum
import random
from collections import namedtuple


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


BLOCK_SIZE = 200
snake_speed = 15

Point = namedtuple('Point', 'x, y')
pygame.init()
# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)
font = pygame.font.SysFont('arial', 25)


class SnakeEnvDir(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 40,
                }
    def __init__(self, action_map=None,
                 env_config=None,
                 render_mode=None
                 ):

        # super(SnakeEnv, self).__init__()

        if env_config is None:
            env_config = dict({"gs": (20, 20),
                               "BLOCK_SIZE":20})


        self.BLOCK_SIZE = env_config["BLOCK_SIZE"]
        self.n_rows = env_config["gs"][0]
        self.n_cols = env_config["gs"][1]


        self.w = self.n_rows * self.BLOCK_SIZE
        self.h = self.n_cols * self.BLOCK_SIZE






        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode,
                                 self._render)

        self.screen = None
        self.clock = None
        self.isopen = True



        self.action_map = {
            0: 'straight',
            1: 'right',
            2: 'left',

        }

        if action_map is not None:
            self.action_map = action_map

        self.action_space = gym.spaces.Discrete(len(self.action_map.keys()))

        self.observation_space = gym.spaces.Box(
            low=-100, high=self.n_rows*self.n_cols, shape=(self.n_rows, self.n_cols, 1),
            dtype=np.int)


        self.window = None
        self.clock = None
        self.high_score = 0
        
        self.reset()




    def reset(self,seed = None):
        # reset the environment to initial state
        # initial direction
        self.direction = Direction.RIGHT


        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y), ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_observation()

    def calc_distance_food(self):
        distance_to_fruit = np.sqrt((self.food.x - self.snake[0].x) ** 2 +
                                    (self.food.y - self.snake[0].y) ** 2)
        self.food_distance = distance_to_fruit


    def get_observation(self):
        # reset state
        self.state = np.zeros((self.n_rows, self.n_cols, 1),
                              dtype=float)
        self.calc_distance_food()
        self.set_borders()
        # update the state (array)
        # put snake in array (0th channel)
        for _i,pt in enumerate(reversed(self.snake)):
            # self.state[int(pt.y // self.n_rows), int(pt.x // self.n_cols), 0] = .5
            self.state[int(pt.y /self.BLOCK_SIZE),
                       int(pt.x / self.BLOCK_SIZE), 0] = .5
        #set Head to 0
        self.state[int(pt.y / self.BLOCK_SIZE), int(pt.x / self.BLOCK_SIZE), 0] = .75

        # place Food
        self.state[int(self.food.y / self.BLOCK_SIZE),
                   int(self.food.x / self.BLOCK_SIZE), 0] = -1

        self.info = dict({"score":self.score,
                          "food_distance":self.food_distance})
        return self.state

    def set_borders(self):

        self.state[:,0,0] = 1
        self.state[0,:,0] = 1
        self.state[:,self.n_cols-1,0] = 1
        self.state[self.n_rows-1,:,0] = 1
        # for i in range(self.n_rows):
        #     self.state[i, 0, 0] = 1.0
        #     self.state[i, self.n_cols, 0] = 1.0
        # for i in range(self.n_cols):
        #     self.n_rows[0, i, 0] = 1.0
        #     self.n_rows[self.n_rows + 1, i, 0] = 1.0


    def plot_state(self,show = True):
        masked_array = np.ma.masked_where(self.state[:, :, 0] == 0, self.state[:, :, 0])

        cmap = matplotlib.cm.jet  # Can be any colormap that you want after the cm
        cmap.set_bad(color='white')
        fig, ax = plt.subplots(1)
        ax.pcolormesh(np.flip(masked_array,axis = 0), cmap=cmap,edgecolor = "black")
        ax.axis("equal")
        ax.axis("off")
        ax.text(0,-1,"Reward: {:.1f}".format(self.reward))
        if show:
            plt.show()
            
    def step(self, action:int):
        # perform one step in the game logic
        # self.frame_iteration += 1
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()




        # print(action)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"


            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.key == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:s
            #         self.direction = Direction.DOWN
        food_distance_old = self.food_distance.copy()
        self._move(action)


        self.snake.insert(0, self.head)
        self.get_observation()

        game_over = False
        # reward = -self.food_distance/(self.w * np.sqrt(2))
        reward = 0
        reward = 0.1 if self.food_distance < food_distance_old else -.1
        self.reward = reward
        if self.is_collision() or self.frame_iteration > (25 * len(self.snake)):
            game_over = True
            reward = -100
            self.get_observation()
            self.reward = reward

            # self.plot_state()

            return self.state, reward, game_over, False, self.info

        # Move head
        if self.head == self.food:
            reward = 100
            self.reward = reward
            # add score
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
            self._place_food()
        else:
            self.snake.pop()

        # self._update_ui()
        # self.clock.tick(snake_speed)
        self.state = self.get_observation()

        self.info = dict({"score":self.score})
        self.renderer.render_step()
        self.reward = reward
        # self.plot_state()
        return self.state, reward, game_over, False, self.info,
        # return observation, reward, done, info

    # def _update_ui(self):
    #     self.display.fill(black)
    #
    #     for pt in self.snake:
    #         pygame.draw.rect(self.display,green,pygame.Rect(pt.x,pt.y,self.BLOCK_SIZE,self.BLOCK_SIZE))
    #     pygame.draw.rect(self.display,white,pygame.Rect(self.food.x,self.food.y,self.BLOCK_SIZE,self.BLOCK_SIZE))
    #
    #     text = self.font.render("Score: " + str(self.score) +" High score: " + str(self.high_score), True, white)
    #     self.display.blit(text, [0, 0])
    #     pygame.display.flip()
    #

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.w, self.h)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.w, self.h))
            if self.clock is None:
                self.clock = pygame.time.Clock()


        # self.surf = pygame.Surface((self.w, self.h))


        self.screen.fill(black)

        #draw border
        #lower border
        pygame.draw.rect(self.screen, red,
                         pygame.Rect(0, 0, self.w, self.BLOCK_SIZE))
        pygame.draw.rect(self.screen, red,
                         pygame.Rect(0, 0, self.BLOCK_SIZE, self.h))
        pygame.draw.rect(self.screen, red,
                         pygame.Rect(self.w-self.BLOCK_SIZE, 0, self.BLOCK_SIZE, self.h))
        pygame.draw.rect(self.screen, red,
                         pygame.Rect(0, self.h-self.BLOCK_SIZE,self.w , self.BLOCK_SIZE))

        #draw snake
        pygame.draw.rect(self.screen, red,
                         pygame.Rect(self.snake[0].x, self.snake[0].y, self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(self.screen, black,
                         pygame.Rect(self.snake[0].x + 5, self.snake[0].y + 5, self.BLOCK_SIZE - 10,
                                     self.BLOCK_SIZE - 10))

        for pt in self.snake[1:]:
            pygame.draw.rect(self.screen, green,
                             pygame.Rect(pt.x + 2, pt.y + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4))
            pygame.draw.rect(self.screen, black,
                             pygame.Rect(pt.x + 5, pt.y + 5, self.BLOCK_SIZE - 10, self.BLOCK_SIZE - 10))
            pygame.draw.circle(self.screen, red, (pt.x + 10,
                                                   pt.y + 10), 2, 2)
        pygame.draw.rect(self.screen, white, pygame.Rect(self.food.x, self.food.y, self.BLOCK_SIZE, self.BLOCK_SIZE))

        text = font.render("Score: " + str(self.score) + " High score: " + str(self.high_score), True, white)

        self.screen.blit(text, [0, 0])
        # self.screen.blit(self.surf, (0,0))


        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )




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

    def _move(self, action):

        clock_wise = [Direction.RIGHT,
                      Direction.DOWN,
                      Direction.LEFT,
                      Direction.UP]

        if self.direction == Direction.DOWN: # down
            if action == 0:
                new_dir = Direction.DOWN
            elif action == 1:
                new_dir = Direction.LEFT
            else:
                new_dir = Direction.RIGHT

        if self.direction == Direction.UP: # up
            if action == 0:
                new_dir = Direction.UP
            elif action == 1:
                new_dir = Direction.RIGHT
            else:
                new_dir = Direction.LEFT

        if self.direction == Direction.LEFT:
            if action == 0:
                new_dir = Direction.LEFT
            elif action == 1:
                new_dir = Direction.UP
            else:
                new_dir = Direction.DOWN

        if self.direction == Direction.RIGHT:
            if action == 0:
                new_dir = Direction.RIGHT
            elif action == 1:
                new_dir = Direction.DOWN
            else:
                new_dir = Direction.UP


        # if np.array_equal(action, [1, 0, 0]):
        #     # keep direction
        #     new_dir = clock_wise[idx]
        # elif np.array_equal(action, [0, 1, 0]):
        #     next_idx = (idx + 1) % 4
        #     new_dir = clock_wise[next_idx]
        # else:
        #     next_idx = (idx - 1) % 4
        #     new_dir = clock_wise[next_idx]
        #
        #
        #
        # new_dir = clock_wise[action]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += self.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.BLOCK_SIZE

        self.head = Point(x, y)
