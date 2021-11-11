import time

import numpy as np
import retro


class BoxingMA:
    def __init__(self, name='Boxing-Atari2600', frameskip=4, framestack=2, render=False):
        self.name = name
        self.env = retro.RetroEnv(name, players=2, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE)

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
        self.state_dim_base = len(self.indexes)

        self.action_dim = self.env.action_space.n
        print(self.action_dim, self.env.observation_space.shape)
        self.framestack = framestack
        self.frameskip = frameskip
        self.do_render = render

        self.minute = 16
        self.seconds = 17
        self.frames = 20

        self.state = np.zeros(self.state_dim_base * framestack)
        self.opp_state = np.zeros_like(self.state)
        self.state[:] = np.tile(self.preprocess(self.env.reset()), framestack)
        self.opp_state[:] = self.state
        self.state_dim = len(self.state)

        permuts = [2,3,0,1,5,4,8,9,6,7,12,13,10,11,14]
        self.index_permut = np.array([x+i*self.state_dim_base for x in permuts for i in range(framestack)])

        # self.action = np.zeros(self.action_dim*2, dtype=np.int16)

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

    def update_opp_state(self):
        self.opp_state[self.index_permut] = self.state


    def step(self, actions):
        for _ in range(self.frameskip):
            observation, _, done, _ = self.env.step(
                actions)
        observation = self.preprocess(observation)
        win = self.win(done, observation)
        self.state[self.state_dim_base:] = self.state[:-self.state_dim_base]
        self.state[:self.state_dim_base] = observation
        self.update_opp_state()

        return done, win

    def reset(self):
        self.state[:] = np.tile(self.preprocess(self.env.reset()), self.framestack)
        self.update_opp_state()

    def render(self):
        time.sleep(0.04)
        self.env.render()
