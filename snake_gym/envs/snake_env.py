import pygame
import numpy as np
import gym
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer
from enum import Enum
import random
from collections import namedtuple


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


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 4,
                }
    def __init__(self, action_map=None,
                 env_config=None,
                 render_mode=None
                 ):

        # super(SnakeEnv, self).__init__()

        if env_config is None:
            env_config = dict({"BLOCK_SIZE": 20,
                               "w": 640,
                               "h": 480})


        self.BLOCK_SIZE = env_config["BLOCK_SIZE"]
        self.w = env_config["w"]
        self.h = env_config["h"]

        self.n_rows = self.h // self.BLOCK_SIZE
        self.n_cols = self.w // self.BLOCK_SIZE


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
            low=0, high=255, shape=(self.n_rows, self.n_cols, 1),
            dtype=np.uint8)


        self.window = None
        self.clock = None
        self.high_score = 0




    def reset(self,seed = None):
        # reset the environment to initial state
        # initial direction
        self.direction = Direction.RIGHT
        self.last_dir = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * self.BLOCK_SIZE), self.head.y), ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_observation()

    def get_observation(self):
        # reset state
        self.state = np.zeros((self.n_rows, self.n_cols, 1),
                              dtype=np.uint8)
        # self.set_borders()
        # update the state (array)
        # put snake in array (0th channel)
        for _i,pt in enumerate(self.snake):
            if _i == 0:
                val = 5
            else:
                val = 0
            self.state[int(pt.y // self.n_rows + 1), int(pt.x // self.n_cols + 1), 0] = val

        # place Food
        self.state[int(self.food.y // self.n_rows + 1), int(self.food.x // self.n_cols + 1), 0] = -1.0
        return self.state

    def step(self, action:int):
        # perform one step in the game logic
        # self.frame_iteration += 1
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()

        #what to do if noob:
        if action == 4:
            action = self.direction.value


        print(action)
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
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
        self.snake.insert(0, self.head)

        game_over = False
        reward = 0
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            self.get_observation()

            return self.state, reward, game_over, False, self.info

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

        # self._update_ui()
        # self.clock.tick(snake_speed)
        self.state = self.get_observation()

        self.info = dict({"score":self.score})
        self.renderer.render_step()

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

        idx = clock_wise.index(self.direction)

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
