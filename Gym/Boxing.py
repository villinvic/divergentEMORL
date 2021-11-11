import time

import gym
import numpy as np


# import retro
# 'Boxing-Atari2600'

class Boxing:
    def __init__(self, name='Boxing-ramNoFrameskip-v4', frameskip=4, framestack=2, render=False):
        self.name = name
        self.env = gym.make(name)
        # self.env = retro.RetroEnv(name, players=2, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE)

        self.ram_locations = dict(player_x=32,
                                  player_y=34,
                                  enemy_x=33,
                                  enemy_y=35,
                                  player_score=18,
                                  enemy_score=19,
                                  player_cd_left=57,
                                  player_cd_right=55,
                                  enemy_cd_left=59,
                                  enemy_cd_right=61,
                                  player_anim_left=75,
                                  player_anim_right=73,
                                  enemy_anim_left=77,
                                  enemy_anim_right=79,
                                  clock=0)

        self.indexes = np.array([value for value in self.ram_locations.values()], dtype=np.int32)
        self.centers = np.array([55, 45, 55, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.scales = np.array(
            [0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.014, 0.014, 0.014, 0.004, 0.004, 0.004, 0.004, 0.004, 0.5],
            dtype=np.float32)


        # 'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT'
        self.action_dim = 10
        self.state_dim_base = len(self.indexes)
        self.state_dim_actions = len(self.indexes) + self.action_dim
        print(self.action_dim, self.env.observation_space.shape)
        self.framestack = framestack
        self.frameskip = frameskip
        self.do_render = render

        self.minute = 16
        self.seconds = 17
        self.frames = 20

        self.state = np.zeros(self.state_dim_actions * framestack)
        self.action_embedding = np.zeros(self.action_dim, dtype=np.float32)

        self.start_state = np.concatenate([self.preprocess(self.env.reset()), self.action_embedding])
        self.state[:] = np.tile(self.start_state, framestack)
        self.state_dim = len(self.state)

        self.past_action = 0

        # self.action = np.zeros(self.action_dim*2, dtype=np.int16)

    def action_to_id(self, action_id):
        self.past_action = action_id
        return action_id  # [action_id, np.random.randint(0, self.action_dim)]

    @staticmethod
    def interpret_hex_as_dec(value):
        as_hex = hex(value)
        try:
            return np.float32(as_hex[2:])
        except ValueError:
            return 0.

    def preprocess(self, obs):
        minutes = np.float32(obs[self.minute] == 0x1b)
        seconds = self.interpret_hex_as_dec(obs[self.seconds])
        frames = obs[self.frames]
        time = minutes + (seconds + frames/60.) / 60.
        obs[0] = time

        return (obs[self.indexes] - self.centers) * self.scales

    def win(self, done, obs):
        if done:
            return np.sign(obs[4] - obs[5])

        return 0

    def step(self, action):
        reward = 0
        for _ in range(self.frameskip):
            observation, rr, done, info = self.env.step(
                self.action_to_id(action))
            reward += rr
        observation = self.preprocess(observation)
        win = self.win(done, observation)
        self.state[self.state_dim_actions:] = self.state[:-self.state_dim_actions]
        self.state[:self.state_dim_base] = observation
        self.state[self.state_dim_base:self.state_dim_base] = 0.
        self.state[self.state_dim_base:self.state_dim_base+self.past_action] = 1.

        return done, win

    def reset(self):
        self.state[:] = np.tile(self.start_state, self.framestack)
        self.past_action = 0

    def render(self):
        time.sleep(0.04)
        self.env.render()
