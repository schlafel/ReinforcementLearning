from model import A2CAgent
import gym
import snake_gymDirected


class SnakeLearningEnv():
    def __init__(self,gs = (12,12)):
        self.input_shape = gs
        self.env = gym.make("SnakeDir-v0", env_config={"gs": gs,
                                                  "BLOCK_SIZE": 20},)


        self.a2_ag = A2CAgent(number_actions = self.env.action_space.n,
                              input_shape = gs)
        self.initialize_models()


    def initialize_models(self):
        state0 = self.env.reset()
        #get summaries
        self.a2_ag.actor.predict(state0[None, :, :, :])
        self.a2_ag.actor.summary()

        self.a2_ag.critic.predict(state0[None, :, :, :])
        self.a2_ag.critic.summary()

        print(30*"*", "Actor ", 30*"*")
        self.a2_ag.actor.summary()

        print(30*"*", "Critic ", 30*"*")
        self.a2_ag.critic.summary()



    def train(self,steps = 3000):

        for s in range(steps):

            done = False
            state = self.env.reset()
            total_reward = 0
            all_aloss = []
            all_closs = []

            while not done:
                # env.render()
                action = self.a2_ag.act(state)
                # print(action)
                next_state, reward, done, _ = self.env.step(action)
                aloss, closs = self.a2_ag.learn(state, action, reward, next_state, done)
                all_aloss.append(aloss)
                all_closs.append(closs)
                state = next_state
                total_reward += reward

                if done:
                    # print("total step for this episord are {}".format(t))
                    print("total reward after {} steps is {}".format(s, total_reward))


def main():
    learner = SnakeLearningEnv()
    learner.train(steps = 3000)



if __name__ == '__main__':

    # https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-1-of-2-d1e26a54ce97
    # https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-2of-2-b8ceb7e059db

    main()
