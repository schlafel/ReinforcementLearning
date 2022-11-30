import keras.models
import os
import gym
import os
import datetime
from statistics import mean
from gym import wrappers
import snake_gym
import tensorflow as tf

if __name__ == '__main__':

    path_model = r"C:\Users\fs.GUNDP\Python\CAS_AML-M3-Project\SNAKE_ENVAVARE_DQN\logs\dqn_snake\20221130-112952\DQN_Snake_0"


    model2 = keras.models.load_model(path_model)

    env = gym.make("Snake-v0", env_config={"w": 400,
                                           "h": 400,
                                           "BLOCK_SIZE": 20},
                   )
    state = env.reset()
    done = False

    while True:
        #now act
        action = tf.math.argmax(model2(state[None,:,:,:]),axis = 1).numpy()[0]
        state, reward, done, _ = env.step(action)
        env.render()

        if done:
            env.reset()
        print("done")
