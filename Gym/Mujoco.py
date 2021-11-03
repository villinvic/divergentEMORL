import gym
import numpy as np
import pprint

def main():
    env = gym.make('Ant-v3')

    done = False
    s = env.reset()
    maxes = np.zeros_like(s)

    for _ in range(10000):
        #env.render()
        print(env.action_space.sample())
        obs,r,done, info = env.step(env.action_space.sample())
        maxes = np.maximum(maxes, np.abs(obs))
        #pprint.pprint(maxes)
        if done:
            env.reset()

    env.close()


class MujocoEnv:

    scales = {'Ant-v3': 1. / np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,
        0.6703889 ,  1.33679745,  0.67496893,  1.35023712,  0.66730872,
        1.36322715,  0.67233852,  1.37066749,  2.87640343,  3.28291404,
        4.3975964 ,  7.43245198,  8.59667592,  6.83814425, 16.08576352,
       17., 17., 17., 17., 17.,
       17. , 17.,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ])
    }

    def __init__(self, name='Ant-v3', render=False):
        self.env = gym.make(name)
        self.render = render
        self.scales = self.scales[name]
        self.obs = self.env.reset() * self.scales
        self.n_rewards = 10

    def step(self, action):
        obs, perf, done, info = self.env.step(action)
        if self.render:
            self.env.render()

        self.obs = obs * self.scales
        info.pop('TimeLimit.truncated', None)
        return perf, done, np.array(list(info.values()))

    def reset(self):
        self.env.reset()

if __name__ == '__main__':
    main()