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


class SnakeEnvV0(gym.Env):
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
                               "BLOCK_SIZE":20,
                               "snake_length":2})


        self.snake_length = env_config["snake_length"]
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
            0: 'right',
            1: 'down',
            2: 'left',
            3: 'up',
            4:'None'
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
        
        # self.reset()

    def reset(self,seed = None):
        # reset the environment to initial state
        # initial direction
        self.state = None
        self.direction = Direction.RIGHT
        self.game_over = False

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      #Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      #Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y),
                      #Point(self.head.x - (3 * self.BLOCK_SIZE), self.head.y),
                      ]
        self.snake.extend(
            [Point(self.head.x - ((n + 1) * self.BLOCK_SIZE), self.head.y) for n in range(self.snake_length)])

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.step_without_scoring = 0
        self.food_distance = self.calc_distance_food()
        return self.get_observation()



    def get_observation(self):
        # reset state
        self.state = np.zeros((self.n_rows, self.n_cols, 2),
                              dtype=float)
        self.calc_distance_food()
        # update the state (array)
        # put snake in array (0th channel)
        for _i,pt in enumerate(reversed(self.snake)):
            # self.state[int(pt.y // self.n_rows), int(pt.x // self.n_cols), 0] = .5
            if not self.game_over:
                self.state[int(pt.y /self.BLOCK_SIZE),
                           int(pt.x / self.BLOCK_SIZE), 0] = 1
        #plot also snake head...
        #set Head to 0
        if not self.game_over:
            self.state[int(pt.y / self.BLOCK_SIZE), int(pt.x / self.BLOCK_SIZE), 1] = 1

        # place Food
        self.state[int(self.food.y / self.BLOCK_SIZE),
                   int(self.food.x / self.BLOCK_SIZE), 0] = -1

        self.info = dict({"score":self.score,
                          "food_distance":self.food_distance})
        return self.state


    def plot_state(self,show = True,actions = None):
        if self.state.shape[0] == 1:
            plot_state = self.state[0, :, :]
        elif self.state.shape[0] == 2:
            plot_state = self.state[0, :, :]
        elif self.state.shape[2] > 1:
            plot_state = self.state[:, :, 0] + self.state[:, :, 1] * -2
        else:
            plot_state = self.state[:, :, 0]
        masked_array = np.ma.masked_where(plot_state == 0, plot_state)
        cmap = matplotlib.cm.jet  # Can be any colormap that you want after the cm
        cmap.set_bad(color='white')
        fig, ax = plt.subplots(1)
        ax.matshow(masked_array, cmap=cmap)
        ax.axis("equal")
        ax.axis("off")

        for (i, j), z in np.ndenumerate(masked_array):
            ax.text(j, i, '{:.0f}'.format(z), ha='center', va='center', fontsize=6,
                    color="white" if z < -1 else "black")
        dirs = ["Right", "Down", "Left", "Up"]
        if actions is not None:
            for _i, val in enumerate(actions.cpu().numpy().flatten()):
                ax.text(-4.5, -.5 + 1 * _i, dirs[_i] + ": {:.2f}".format(val),
                        weight="bold" if val == max(actions.cpu().numpy().flatten()) else "normal")

        plt.tight_layout()
        if show:
            plt.show()
            
    def step(self, action:int):
        # perform one step in the game logic

        self.direction_old = self.direction
        #what to do if noob:
        if action == 4:
            action = self.direction.value

        # print(action)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        self._move(action)
        self.snake.insert(0, self.head)

        game_over = False
        self.game_over = game_over
        reward = 0
        if self.is_collision() or self.frame_iteration > (25 * len(self.snake)):
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
            self.frame_iteration = 0
        else:
            self.snake.pop()

        self.state = self.get_observation()
        self.info = dict({"score":self.score})

        if self.screen is not None:
            self.renderer.render_step()

        self.frame_iteration+=1
        return self.state, reward, game_over, False, self.info,

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


        self.screen.fill(black)

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
        if pt.x > (self.w - self.BLOCK_SIZE) or pt.x < 0 or pt.y > (self.h - self.BLOCK_SIZE) or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _place_food(self):
        x = random.randint(0, (self.w - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
        y = random.randint(0, (self.h - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _move(self, action):

        clock_wise = [Direction.RIGHT,
                      Direction.DOWN,
                      Direction.LEFT,
                      Direction.UP]

        new_dir = clock_wise[action]
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

    def calc_distance_food(self):
        distance_to_fruit = np.sqrt((self.food.x - self.snake[0].x) ** 2 +
                                    (self.food.y - self.snake[0].y) ** 2)
        self.food_distance = distance_to_fruit
        return self.food_distance

class SnakeEnvV1(SnakeEnvV0):
    """

    """
    def __init__(self,action_map=None,
                 env_config=None,
                 render_mode=None):
        super().__init__(action_map=action_map,
                 env_config=env_config,
                 render_mode=None)

        self.action_map = {
            0: 'straight',
            1: 'right',
            2: 'left',

        }

        self.action_space = gym.spaces.Discrete(len(self.action_map.keys()))

    def _move(self, action):

        if self.direction == Direction.DOWN: # downsdss
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
class SnakeEnvV2(SnakeEnvV1):
    #Agent with Borders
    def __init__(self,action_map=None,
                 env_config=None,
                 render_mode=None):
        super().__init__(action_map=action_map,
                 env_config=env_config,
                 render_mode=None)



    def get_observation(self):
        # reset state
        self.state = np.zeros((self.n_rows, self.n_cols, 2),
                              dtype=float)
        self.calc_distance_food()
        self.set_borders()
        # update the state (array)
        # put snake in array (0th channel)
        for _i,pt in enumerate(reversed(self.snake)):
            # self.state[int(pt.y // self.n_rows), int(pt.x // self.n_cols), 0] = .5
            self.state[int(pt.y /self.BLOCK_SIZE),
                       int(pt.x / self.BLOCK_SIZE), 0] = 1
        #set Head to 0
        self.state[int(pt.y / self.BLOCK_SIZE), int(pt.x / self.BLOCK_SIZE), 1] = 1

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















        
class SnakeEnvVanilla(SnakeEnvV1):

    def __init__(self,action_map=None,
                 env_config=None,
                 render_mode=None):
        super().__init__(action_map=action_map,
                 env_config=env_config,
                 render_mode=None)

        self.action_map = {
            0: 'straight',
            1: 'right',
            2: 'left',        }

        self.action_space = gym.spaces.Discrete(len(self.action_map.keys()))
        self.observation_space = gym.spaces.Discrete(11)
    
    def get_observation(self):
        ## 11 Values
        #[danger_straight, danger_right,danger_left,danger_right
        head = self.snake[0]
        bs = self.BLOCK_SIZE
        point_l = Point(head.x - bs,head.y)
        point_r = Point(head.x + bs,head.y)
        point_u = Point(head.x,head.y-bs)
        point_d = Point(head.x,head.y+bs)


        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]

        return np.array(state,dtype = int)