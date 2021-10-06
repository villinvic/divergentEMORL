import numpy as np

class Rewards:
    base = {
        'velocity': 0.5,
        'movement': 5.,
        'win': 1.,
    }

    special = np.array([
    ])

    main = np.array([
        'win'
    ])


    def __init__(self, batch_size, trajectory_length, area_size, n_cyclones):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

        self.n_cyclones = n_cyclones
        self.area_size = area_size

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute(self, states, reward_shape, base_rewards):

        self['velocity'] = np.sqrt((states[:, 1:, 6 * self.n_cyclones + 2] - states[:, :-1, 6 * self.n_cyclones + 2])**2 + \
                           (states[:, 1:, 6 * self.n_cyclones + 3] - states[:, :-1, 6 * self.n_cyclones + 3])**2)

        last_dist = np.sqrt((states[:, 1:, 6 * self.n_cyclones] ** 2 + states[:, 1:, 6 * self.n_cyclones + 1]) ** 2)
        self['movement'] = last_dist - \
                           np.sqrt((states[:, :-1, 6 * self.n_cyclones] ** 2 + states[:, :-1, 6 * self.n_cyclones + 1]) ** 2)

        self['win'][:, :] = base_rewards

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)
        #self.values[:, :] = (1.0 - reward_shape['negative_scale']) * np.maximum(total, 0.) + total

        return self.values




