import numpy as np


class TennisRewards:
    base = {
        'movement': 0.01,
        #'back': 0.01,
        #'front': 0.01,
        'aim': 1.,
        'score': 1.,
    }

    special = np.array([
    ])

    main = 'score'


    def __init__(self, batch_size, trajectory_length):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)
        self.is_returning = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute(self, states, reward_shape, base_rewards):

        """
            def self_dy(self, full_obs):
        dx = np.abs(full_obs[5] - full_obs[-self.state_dim + 5])
        dy = np.abs(full_obs[6] - full_obs[-self.state_dim + 6])
        d = dx + dy
        if d > 30 * 0.01:
            dy = 0
        return dy

        """
        self['movement'] = np.sqrt((states[:, 1:, 5] - states[:, :-1, 5-20])**2 + (states[:, 1:, 6] - states[:, :-1, 6-20])**2)

        #self['opp_movement'] = np.clip(
        #    np.sqrt((states[:, :-1, 0] - states[:, 1:, 0]) ** 2 + (states[:, :-1, 1] - states[:, 1:, 1]) ** 2), 0, 1.5)
        #             return obs[6] > 1.022
        #         else:
        #             return obs[6] < 0.028
        #self['back'] = -np.float32( np.logical_or(states[:, 1:, 6]<=0.0161, states[:, 1:, 6]>=1.015))
        #         if self.side:
        #             return obs[6] < 0.756
        #         else:
        #             return obs[6] > 0.294
        #self['front'] = -np.float32( np.logical_and(states[:, 1:, 6]<0.693, states[:, 1:, 6]>0.616))

        # np.abs(preprocessed_obs[4]-preprocessed_obs[6]) < 0.25 and preprocessed_obs[9] < 0.016 and np.abs(preprocessed_obs[9] - preprocessed_obs[9 + 21]) > 1e-5
        self['aim'] = np.float32(np.logical_and(np.logical_and(np.logical_and(states[:, 1:, 8]!=states[:, :-1, 8], states[:, 1:, 8] == states[:, 1:, 9]),
                                                np.abs(states[:, 1:, 4] - states[:, 1:, 6]) < 30 * 0.007),
                                 np.abs(states[:, 1:, 3] - states[:, 1:, 5]) < 30 * 0.007))

        self['score'] = base_rewards
        self['efficiency'] = - np.float32(np.logical_and(np.logical_and(np.logical_and(states[:, 1:, 8]!=states[:, :-1, 8], (1.-states[:, 1:, 8]) == states[:, 1:, 9]),
                                                np.abs(states[:, 1:, 4] - states[:, 1:, 2]) < 30 * 0.007),
                                 np.abs(states[:, 1:, 3] - states[:, 1:, 1]) < 30 * 0.007))

        print(np.mean(self['aim']))

        all_wins = np.sum(np.clip(base_rewards, 0., 1.))
        all_points = np.sum(np.abs(base_rewards))
        if all_points == 0:
            performance = np.nan
        else:
            performance = all_wins / all_points * 100 - 50

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)

        return self.values, performance



class BoxingRewards:

    base = {
#        'movement': 0.05,
        'hit': 0.05,
        'hurt': 0.05,
        'win': 1.,
#        'away': 0.05,
    }

    main = 'win'

    def __init__(self, batch_size, trajectory_length):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute_damage(self, obs):
        injury = obs[5 + self.state_dim] - obs[5]
        damage = obs[4 + self.state_dim] - obs[4]

        return np.clip(damage / self.scales[4], 0, 2), np.clip(injury / self.scales[5], 0, 2)

    def compute(self, states, reward_shape, base_rewards):

        #self['movement'][:] = np.clip(np.sqrt((states[:, :-1, 0 + len(states)//2]-states[:, 1:, 0])**2+(states[:, :-1, 1 + len(states)//2]-states[:, 1:, 1])**2), 0, 1.5)
        #self['away'][:] = -np.clip(np.sqrt((states[:, :-1, 0] - states[:, :-1, 2]) ** 2 + (states[:, :-1, 1] - states[:, :-1, 3]) ** 2) - \
        #    np.sqrt((states[:, 1:, 0] - states[:, 1:, 2]) ** 2 + (states[:, 1:, 1] - states[:, 1:, 3]) ** 2), 0, 1.5)

        self['hit'][:] = np.clip( states[:, 1:, 4]- states[:, :-1, 4], 0., np.inf)
        self['hurt'][:] = -np.clip(states[:, 1:, 5] - states[:, :-1, 5], 0, np.inf)

        self['win'][:] = base_rewards

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)

        all_wins = np.sum(np.clip(base_rewards, 0., 1.))
        all_points = np.sum(np.abs(base_rewards))
        if all_points == 0:
            performance = np.nan
        else:
            performance = all_wins/all_points * 100  - 50

        print(performance)

        return self.values, performance


class KfmRewards:

    base = {
        'movement': 0.05,
        'jump': 0.05,
        'hurt': 0.05,
        'score': 0.05,
        'death': 1.0,
    }

    main = 'win'

    def __init__(self, batch_size, trajectory_length):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]


    def compute(self, states, reward_shape, base_rewards):

        self['movement'][:] = np.abs(states[:, :-1, 74] - states[:, 1:, 74])
        self['jump'][:] = np.abs(states[:, 1:, 74] - states[:, :-1, 74])
        self['hurt'][:] = -np.clip(states[:, :-1, 28] - states[:, 1:, 28], 0, np.inf)
        self['score'][:] = np.clip(states[:, 1:, 25] - states[:, :-1, 25], 0, np.inf)
        self['death'][:] = np.float32(np.logical_and(states[:, 1:, 28] < 1e-4, states[:, :-1, 28] > 1e-4))

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)

        return self.values, np.mean(base_rewards[np.logical_not(np.isnan(base_rewards))]) * 255


class MujocoRewards:
    indexes = {
        'reward_forward': 0,
        'reward_ctrl': 1,
        'reward_contact': 2,
        'reward_survive': 3,
        'x_position': 4,
        'y_position': 5,
        'distance_from_origin': 6,
        'x_velocity': 7,
        'y_velocity': 8,
        'forward_reward': 9}

    rewards = [
        'reward_forward',
        'reward_ctrl',
        'reward_contact',
        'reward_survive'
    ]

    base = {
        'reward_forward'      : 1.0,
        'reward_ctrl'         : 1.0,
        'reward_contact'      : 1.0,
        'reward_survive'      : 1.0,
        'velocity'            : 1.0,
        'perf'                : 1.0,
    }

    def __init__(self, batch_size, trajectory_length):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute(self, rewards, reward_shape, perf):

        for reward_name in self.rewards:
            self[reward_name] = rewards[:, :, self.indexes[reward_name]]

        self['velocity'] = rewards[:, :, self.indexes['x_velocity']]+rewards[:, :, self.indexes['y_velocity']]

        self['perf'] = perf

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)

        return self.values