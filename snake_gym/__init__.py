import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Snake-v0',
    entry_point='snake_gym.envs:SnakeEnvV0',

)


register(
    id='Snake-v1',
    entry_point='snake_gym.envs:SnakeEnvV1',

)

register(
    id='Snake-v2',
    entry_point='snake_gym.envs:SnakeEnvV2',

)
register(
    id='Snake-Vanilla',
    entry_point='snake_gym.envs:SnakeEnvVanilla',

)
register(
    id='Snake-v0PyTorch',
    entry_point='snake_gym.envs:SnakeEnvV0PyTorch',

)
register(
    id='Snake-v1PyTorch',
    entry_point='snake_gym.envs:SnakeEnvV1PyTorch',

)


