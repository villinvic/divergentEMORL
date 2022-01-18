import gym
import numpy as np
import time

class Tennis:
    def __init__(self, name='Tennis-ramNoFrameskip-v4', frameskip=2, framestack=2, render=False):
        self.name = name
        self.env = gym.make(name)

        self.ram_locations = dict(enemy_x=27,
                                     enemy_y=25,
                                     #enemy_score=70,
                                     ball_x=16,
                                     ball_y=15,
                                     player_x=26,
                                     player_y=24,
                                     #player_score=69,
                                     ball_height=17,
                                     ball_direction=52,
                                     side=80,
                                     ball_bounce_num=76,
                                  )
        # 73 -> ball being hit

        self.indexes = np.array([value for value in self.ram_locations.values()], dtype=np.int32)
        self.reversed_indexes = np.array([26, 24, 16, 15, 27, 25, 17, 52, 80, 76], dtype=np.int32)
        self.centers = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.scales = np.array([0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.025, 1.0, 1., 0.5], dtype=np.float32)
        self.y_bounds = (0.91, 1.48)
        # 2 - 74 75 - 148
        self.side = True
        self.max_frames = 15000
        self.frames_since_point = 0

        self.points = np.array([71, 72], dtype=np.int32)
        self.top_side_points = np.array([0, 3, 4, 7, 8, 11])

        self.action_dim = 9

        self.state_dim_base = len(self.indexes) + 1
        self.state_dim_actions = self.state_dim_base + self.action_dim
        self.framestack = framestack
        self.frameskip = frameskip
        self.do_render = render

        self.state = np.zeros(self.state_dim_actions * framestack)
        self.action_embedding = np.zeros(self.action_dim, dtype=np.float32)

        self.start_state = np.concatenate([self.preprocess(self.env.reset()), self.action_embedding])
        self.state[:] = np.tile(self.start_state, framestack)
        self.state_dim = len(self.state)
        self.past_action = 0

    def action_to_id(self, action_id):
        # ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE',
        # 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']


        if action_id > 0:
            action_id += 9
        else:
            action_id = 1


        return action_id

    def preprocess(self, obs):

        if self.side:
            indexes = self.indexes
        else:
            indexes = self.reversed_indexes

        if obs[self.ram_locations['ball_bounce_num']] > 200:
            obs[self.ram_locations['ball_bounce_num']] = -1
        obs[self.ram_locations['ball_bounce_num']] += 1


        x = (obs[indexes] - self.centers) * self.scales
        return np.concatenate([[np.clip(1. - self.frames_since_point * 0.0005, 0, 1)], x])

    def distance_from_ball(self, obs):
        return np.sqrt((obs[4] - obs[6]) ** 2 + (obs[5] - obs[7]) ** 2) * 0.9

    def is_back(self, obs):
        # print(np.sqrt((obs[5]-obs[3])**2+(obs[6]-obs[4])**2))
        # for ob, k in zip(obs, self['ram_locations'].keys()):
        #    print(k, ob)
        # print()
        if self.side:
            return obs[6] > 1.022
        else:
            return obs[6] < 0.028

    def is_front(self, obs):
        # print(obs[3]*100, obs[4]*100)
        if self.side:
            return obs[6] < 0.756
        else:
            return obs[6] > 0.294

    def distance_ran(self, obs, obs_):
        d = np.sqrt((obs[1] - obs_[1]) ** 2 + (obs[0] - obs_[0]) ** 2)
        if d > 20 * 0.01:
            d = 0
        return d

    def self_dy(self, full_obs):
        dx = np.abs(full_obs[5] - full_obs[-self.state_dim + 5])
        dy = np.abs(full_obs[6] - full_obs[-self.state_dim + 6])
        d = dx + dy
        if d > 30 * 0.01:
            dy = 0
        return dy

    def aim_quality(self, full_obs):
        ball_x = full_obs[-self.state_dim + 3]
        ball_y = full_obs[-self.state_dim + 4]
        dplayer_y = full_obs[-self.state_dim + 1] - full_obs[-self.state_dim + 6]
        # print('1', ball_x, ball_y)
        # print('2', full_obs[-2*self.state_dim+3], full_obs[-2*self.state_dim+4])
        vector = complex(ball_y - full_obs[-2 * self.state_dim + 4], ball_x - full_obs[-2 * self.state_dim + 3])
        vector_p2p = complex(dplayer_y, full_obs[-self.state_dim + 0] - full_obs[-self.state_dim + 5])
        if self.side:
            vector *= -1
            vector_p2p *= -1
        angle = np.angle(vector)
        angle_2 = np.angle(vector_p2p)
        # print(full_obs[-self.state_dim+1]-full_obs[-self.state_dim+6])

        quality = np.clip((angle - angle_2) ** 2 * 2 + 0.1, 0, 2.25)
        if np.abs(dplayer_y) < 0.35:
            quality *= 0.2

        # opp_x = full_obs[-self.state_dim]
        # opp_y = full_obs[-self.state_dim+1]
        # dY = opp_y - ball_y

        # scale = (0.5 + 0.2 * np.abs(dY)) * np.sign(dY)
        # deviation = np.tan(angle) * scale

        # quality = np.clip(np.abs(ball_x + deviation - opp_x), 0, 1) + 0.2

        return quality

    def proximity_to_front(self, obs):
        if self.side:
            return (np.abs(0.406 - (obs[6] - 0.63)) / 0.406) ** 2
        else:
            return ((obs[6] - 0.014) / 0.406) ** 2

    def proximity_to_back(self, obs):
        if self.side:
            return (np.abs(0.406 - (1.036 - obs[6])) / 0.406) ** 2
        else:
            return (np.abs(0.406 - (obs[6] - 0.014)) / 0.406) ** 2

    def is_returning(self, obs):
        return obs[8]!=obs[8+self.state_dim_actions] and obs[8]==obs[9] and np.abs(obs[4] - obs[6]) < 30 * 0.007 and np.abs(obs[3] - obs[5]) < 30 * 0.007

    def win(self, obs, last_obs):
        dself = obs[7] - last_obs[7]
        dopp = obs[2] - last_obs[2]
        dscore = np.clip(dself, 0, 1) - np.clip(dopp, 0, 1)
        return dscore

    def swap_court(self, full_obs):
        self.side = full_obs[self.ram_locations['side']] < 1

    def step(self, action):
        reward = 0
        for _ in range(self.frameskip):
            observation, rr, done, info = self.env.step(
                self.action_to_id(action))
            reward += rr


        while not done and observation[75] < 255:

            for _ in range(self.frameskip):
                observation, rr, done, info = self.env.step(
                    self.action_to_id(action))
                reward += rr
                if done:
                    break

        self.swap_court(observation)
        observation = self.preprocess(observation)
        #if self.is_returning(observation):

        if reward == 0:
                self.frames_since_point += 1

        self.state[self.state_dim_actions:] = self.state[:-self.state_dim_actions]
        self.state[:self.state_dim_base] = observation
        #print(self.is_returning(self.state), observation[8], observation[9], np.abs(observation[4] - observation[6]),
        #      np.abs(observation[3] - observation[5]))
        self.state[self.state_dim_base:self.state_dim_actions] = 0.
        self.state[self.state_dim_base+self.past_action] = 1.

        return done, reward

    def reset(self):
        self.env.reset()
        self.side = True
        self.frames_since_point = 0
        self.state[:] = np.tile(self.start_state, self.framestack)
        self.past_action = 0

    def render(self):
        time.sleep(0.01)
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


    def locations(self, states):
        return states[:, 5:7]


