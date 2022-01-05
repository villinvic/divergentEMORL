import time

import numpy as np

np.set_printoptions(suppress=True)

import retro
# 'Boxing-Atari2600'

class Kfm:
    def __init__(self, name='KungFuMaster-Atari2600', frameskip=4, framestack=2, render=False):
        self.name = name

        self.env = retro.RetroEnv(name, players=1, obs_type=retro.Observations.RAM)
        self.action_dict = {
            0 : np.array([0,0,0,0,0,0,0,0]),
            1 : np.array([1,0,0,0,0,0,0,0]),
            2 : np.array([0,0,0,0,1,0,0,0]),
            3: np.array([0, 0, 0, 0, 0, 0, 0, 1]),
            4: np.array([0, 0, 0, 0, 0, 0, 1, 0]),
            5: np.array([0, 0, 0, 0, 0, 1, 0, 0]),
            6: np.array([1, 0, 0, 0, 1, 0, 0, 0]),
            7: np.array([1, 0, 0, 0, 0, 0, 0, 1]),
            8: np.array([1, 0, 0, 0, 0, 0, 1, 0]),
            9: np.array([1, 0, 0, 0, 0, 1, 0, 0]),
        }

        """
        player_x = 202 / 255
        player_y = 60 / 30
        player_hp = 156 / 133
        enemy_x = 200 / 255
        projectile_x = 201 / 255
        lives = 29 / 255
        score = 25
        -128
        """
        self.scales = [254., 255.,   1.,   7., 192.,   1,   5., 255.,   1.,   1.,   9.,   1.,   1.,  18.,
                      18.,   1.,   8.,  36., 128.,  60.,   3.,   1.,   1.,   1.,   1.,  100.,   1.,  32.,
                     153., 255.,   3.,   1.,   1.,  31.,  64.,  47., 172., 223., 144., 218., 218., 221.,
                      62., 220., 209., 221., 185., 218., 215., 218., 177., 219.,   3., 255.,  64., 103.,
                      27.,  11., 255., 255.,  25., 128.,  38.,   3.,   9., 255.,  64.,   1.,   2., 128.,
                     209., 209., 208., 207., 208.,  39.,  39., 160.,  64., 255.,   1.,  30.,   1., 251.,
                     251., 251., 250., 251., 245., 165.,   8.,   3.,   8.,   8., 197., 128., 188., 230.,
                     157., 223., 157., 223., 157., 223., 157., 223.,   1., 223., 180., 223., 185., 223.,
                     190., 223., 157., 223., 157., 223., 130., 141., 216.,   1., 165.,  22., 200., 252.,
                     141., 211.]

        self.ram_locations = dict()
        self.action_dim = 10
        self.state_dim_base = 128
        self.state_dim_actions = 128 + self.action_dim
        self.framestack = framestack
        self.frameskip = frameskip
        self.do_render = render

        self.state = np.zeros(self.state_dim_actions * framestack)
        self.action_embedding = np.zeros(self.action_dim, dtype=np.float32)

        s = self.preprocess(self.env.reset())
        self.start_state = np.concatenate([s, self.action_embedding])
        self.state[:] = np.tile(self.start_state, framestack)
        self.state_dim = len(self.state)

        self.past_action = 0




        # self.action = np.zeros(self.action_dim*2, dtype=np.int16)

    def action_to_id(self, action_id):
        self.past_action = action_id
        return self.action_dict[action_id] # [action_id, np.random.randint(0, self.action_dim)]

    def preprocess(self, obs):
        return obs / self.scales

    def win(self, done, obs):
        return obs[25] if done else np.nan

    def step(self, action):
        reward = 0
        for _ in range(self.frameskip):
            observation, rr, done, info = self.env.step(
                self.action_to_id(action))
            reward += rr

        #self.scales[:]= np.maximum(self.scales, observation)
        observation = self.preprocess(observation)
        self.state[self.state_dim_actions:] = self.state[:-self.state_dim_actions]
        self.state[:self.state_dim_base] = observation
        self.state[self.state_dim_base:self.state_dim_actions] = 0.
        self.state[self.state_dim_base+self.past_action] = 1.

        return done, self.win(done, observation)

    def reset(self):
        print(self.scales)
        self.env.reset()
        self.state[:] = np.tile(self.start_state, self.framestack)
        self.past_action = 0

    def render(self):
        time.sleep(0.04)
        self.env.render()

    def compute_stats(self, states, final_states, scores):
        stats = dict()
        stats['win_rate'] = scores[0] / np.float32(np.sum(scores))

        actions = np.sum(states[:, self.state_dim_base:self.state_dim_actions], axis=0)
        stats['action_avg_prob'] = actions / np.sum(actions)
        stats['distance'] = np.mean(np.sqrt((states[:, 0] - states[:, 2])**2 + (states[:, 1] - states[:, 3])**2))
        stats['avg_timer'] = np.mean(states[final_states, 14])
        stats['avg_punches'] = np.mean(states[final_states, 4])
        stats['avg_hurt'] = np.mean(states[final_states, 5])
        stats['mobility'] = np.mean(np.sqrt((states[1:, 0] - states[:-1, 0])**2 + (states[1:, 1] - states[:-1, 1])**2))

        return stats

    def punch_locations(self, states):
        ds = states[1:] - states[:-1]
        return self.locations(states[1:][np.where(ds[:, 4]>0)])

    def locations(self, states):
        return states[:, :2]


if __name__ == '__main__':

    x = Kfm()
    p = np.random.random((10,))
    p /= p.sum()
    for _ in range(1000):
        done, r= x.step(np.random.choice(10, p=p))
        x.render()
        if done:
            x.reset()
            print(r)
    #print(x.state[157-128])
