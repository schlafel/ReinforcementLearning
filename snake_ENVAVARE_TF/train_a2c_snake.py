import os

from model import Model,A2CAgent
import tensorflow as tf
import gym
import snake_gym
import numpy as np
import tqdm
from tensorflow.keras.utils import Progbar

class A2C_Learner:

    def __init__(self,env, num_envs = 1):

        self.env = env

        self.model =  Model()
        self.agent = A2CAgent(self.model)
        self.agent.initialize_model(self.env)

        self.num_envs = 1

    def print_model_summary(self):
        print(self.agent.model.summary())

    def _a(self):
        pass


    def run(self,run_name = "test1",
            show_training = True,
            batch_size = 128):
        if not os.path.exists(run_name):
            os.makedirs(run_name)
        # train model
        rew = np.zeros((self.num_envs, 1))

        #get state
        state = env.reset()

        steps = 0
        num_eps = 0
        steps_since_last_test = 0
        self.epoch = 1
        while True:
            print(" ")
            print(30*"*")
            print("Epoch", self.epoch)
            state, num_steps, num_eps = self.run_train_step(
                state=state,
                rew=rew,
                env=self.env,
                num_actions=4,
                batch_size=batch_size,
                num_steps=steps,
                num_eps=num_eps,
                show_training=show_training,
            )
            steps += num_steps
            steps_since_last_test += num_steps
            if steps_since_last_test >= 500000:
                folder = run_name
                if not os.path.exists(folder):
                    os.mkdir(folder)
                self.model.model.save(f'{folder}/{steps}_main.h5')
                self.model.policy_head.save(f'{folder}/{steps}_policy.h5')
                self.model.value_head.save(f'{folder}/{steps}_value.h5')

            self.epoch+=1

    def run_train_step(self,state,
                       rew, env,num_actions = 4,
                       batch_size = 500,
                       num_steps = 50000,
                       num_eps = 500,
                       show_training = True):

        sarsdv= []
        for i in tqdm.tqdm(range(batch_size)):
            num_steps += num_steps

            action_logits, action_probs, value = self.agent.model(state[None,:])


            action = np.random.choice(range(num_actions), p=action_probs.numpy().flatten())
            next_s, reward, done, info_dict = env.step(action)
            if show_training:
                env.render()
            if self.epoch%500 == 0:
                env.render()
                #env.plot_state()
            next_s = np.stack(next_s)
            reward = np.expand_dims(np.stack([reward]),-1)
            done = np.expand_dims(np.stack([done]), -1)

            rew += reward
            sarsdv.append(([state], [action], reward, None, done, value))

            state = next_s.copy()

            if done:
                num_eps +=1
                state = env.reset()
                rew = 0


        _, _, R = self.agent.model(state[None,:])

        discounted_rewards = []
        for _, _, r, _, d, _ in reversed(sarsdv):
            R = r + self.agent.params["gamma"] * R * (1 - d)
            discounted_rewards.append(R)

        tots = len(sarsdv)

        discounted_rewards = np.concatenate(np.array(list(reversed(discounted_rewards))))

        states = self._a(sarsdv, 0)
        values = self._a(sarsdv, -1)
        actions = self._a(sarsdv, 1)


        loss,pol_loss,val_loss = self.agent.train(states.astype('float32'),
                                discounted_rewards.astype('float32'),
                                values.astype('float32'),
                                actions.astype('int32'))
        print(loss.numpy())
        print("High Score:", env.high_score)
        return state, num_steps, num_eps

    @staticmethod
    def _a(l, idx):
        return np.concatenate([m[idx] for m in l])



if __name__ == '__main__':

    env = gym.make("Snake-v0",env_config = {"w":400,
                                             "h":400,
                                            "BLOCK_SIZE":20},
                   )
    aL = A2C_Learner(env = env)
    #aL.print_model_summary()

    aL.run(run_name="test",
           batch_size=128*16,
           show_training=False)
