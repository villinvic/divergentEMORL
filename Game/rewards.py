import numpy as np

class Rewards:
    base = {
        'velocity': 1.,
        'near_target': 1.,
        'exploration': 1.,
        'time': 1.,
        'win': 1.,
    }

    special = np.array([
    ])

    main = 'win'


    def __init__(self, batch_size, trajectory_length, area_size, max_see, view_range):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

        self.max_see = max_see
        self.area_size = area_size
        self.view_range = view_range

        self.dist = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute(self, states, reward_shape, base_rewards):


        self['time'][:,: ] = -1.0
        self['velocity'] = np.sqrt(states[:, 1:, 6 * self.max_see + 2]**2 + states[:, 1:, 6 * self.max_see + 3]**2) - 0.2

        self.dist[:, :] = states[:, 1:, 4] == 1.
        self['near_target'] = np.float32(self.dist)


        self['win'][:, :] = base_rewards

        self['exploration'] = (-np.sqrt((states[:, 1:, 6 * self.max_see])**2+(states[:, 1:, 6 * self.max_see+1])**2)\
               + np.sqrt((states[:, :-1, 6 * self.max_see])**2+(states[:, :-1, 6 * self.max_see+1])**2))

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)

        performance = np.sum(np.mean(base_rewards/self.base[self.main], axis=0))

        return self.values, performance




