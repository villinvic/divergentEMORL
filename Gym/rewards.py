import numpy as np


class TennisRewards:
    base = {
        'movement': 0.01,
        'opp_movement': 0.01,
        'back': 0.01,
        'front': 0.01,
        'aim': 1.0,
        'score': 0.2,
        'win': 1.,
    }

    special = np.array([
    ])

    main = 'win'


    def __init__(self, state_dim, batch_size, trajectory_length):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

        self.state_dim = state_dim

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

        self['movement'] = np.sqrt((states[:, 1:, 5 - self.state_dim]-states[:, :-1, 5 - self.state_dim])**2 + (states[:, 1:, 6 - self.state_dim]-states[:, :-1, 6 - self.state_dim])**2)
        self['opp_movement'] = np.sqrt((states[:, 1:, - self.state_dim] - states[:, :-1, - self.state_dim]) ** 2 + (
                    states[:, 1:, 1 - self.state_dim] - states[:, :-1, 1 - self.state_dim]) ** 2)
        self.dist[:, :] = states[:, 1:, 4] == 1.
        self['near_target'] = np.float32(self.dist)


        self['win'][:, :] = base_rewards

        self['exploration'] = (-np.sqrt((states[:, 1:, 6 * self.max_see])**2+(states[:, 1:, 6 * self.max_see+1])**2)\
               + np.sqrt((states[:, :-1, 6 * self.max_see])**2+(states[:, :-1, 6 * self.max_see+1])**2))

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)

        performance = np.sum(np.mean(base_rewards/self.base[self.main], axis=0))

        return self.values, performance


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


