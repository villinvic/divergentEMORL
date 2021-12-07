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
        self.index_permut = np.array([2,3,0,1,5,4,8,9,6,7,12,13,10,11,14], dtype=np.int32)
        self.action_dim = 16
        self.state_dim_base = len(self.indexes)
        self.state_dim_actions = len(self.indexes) + self.action_dim

        self.framestack = framestack
        self.frameskip = frameskip
        self.do_render = render

        self.minute = 16
        self.seconds = 17
        self.frames = 20

        self.state = np.zeros(self.state_dim_actions * framestack)
        self.opp_state = np.zeros_like(self.state)

        self.action_embedding = np.zeros((2, self.action_dim), dtype=np.float32)
        self.past_action = np.zeros(2, dtype=np.int32)
        init_state = self.preprocess(self.env.reset())
        opp_init_state = init_state[self.index_permut]
        self.start_state = np.concatenate([init_state, self.action_embedding[0]])
        self.opp_start_state = np.concatenate([opp_init_state, self.action_embedding[1]])

        self.state[:] = np.tile(self.start_state, framestack)

        self.state_dim = len(self.state)

    def action_to_id(self, actions):
        self.past_action[:] = actions
        return actions

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
        obs = obs.astype(np.float32)
        obs[0] = time

        return (obs[self.indexes] - self.centers) * self.scales

    def win(self, done, obs):
        if done:
            return np.sign(obs[4] - obs[5])

        return 0

    def update_opp_state(self):
        self.opp_state[self.index_permut] = self.state[:self.state_dim_base]
        self.opp_state[self.state_dim_base:self.state_dim_actions] = 0.
        self.opp_state[self.state_dim_base + self.past_action[1]] = 1.


    def step(self, actions):
        #reward = 0
        actions = self.action_to_id(actions)
        if self.do_render:
            self.render()
        for _ in range(self.frameskip):
            observation, rr, done, info = self.env.step(actions)
            #reward += rr
        observation = self.preprocess(observation)
        win = self.win(done, observation)
        self.state[self.state_dim_actions:] = self.state[:-self.state_dim_actions]
        self.opp_state[self.state_dim_actions:] = self.opp_state[:-self.state_dim_actions]
        self.state[:self.state_dim_base] = observation
        self.state[self.state_dim_base:self.state_dim_actions] = 0.
        self.state[self.state_dim_base+self.past_action[0]] = 1.
        self.update_opp_state()

        return done, win

    def reset(self):
        self.env.reset()
        self.state[:] = np.tile(self.start_state, self.framestack)
        self.opp_state[:] = np.tile(self.opp_start_state, self.framestack)
        #self.update_opp_state()
        self.past_action = [0,0]

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
