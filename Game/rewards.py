import numpy as np

class Rewards:
    base = {
        'velocity': 1.,
        'movement': 1.,
        'win': 1.,
    }

    special = np.array([
    ])

    main = 'win'


    def __init__(self, batch_size, trajectory_length, area_size, n_cyclones, n_exits):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

        self.n_cyclones = n_cyclones
        self.n_exits = n_exits
        self.n_entities = n_exits + n_cyclones
        self.area_size = area_size

        self.dist = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute(self, states, reward_shape, base_rewards):



        self['velocity'] = ( np.abs(states[:, 1:, 6 * self.n_entities + 2]) - np.abs(states[:, :-1, 6 * self.n_entities + 2])+
                           np.abs(states[:, 1:, 6 * self.n_entities + 3]) - np.abs(states[:, :-1, 6 * self.n_entities + 3]))

        self.dist[:, :] = (np.sqrt((states[:, 1:, 6 * self.n_entities]-states[:, 1:, 0])**2+(states[:, 1:, 6 * self.n_entities+1]-states[:, 1:, 1])**2))\
               - (np.sqrt((states[:, :-1, 6 * self.n_entities]-states[:, :-1, 0])**2+(states[:, :-1, 6 * self.n_entities+1]-states[:, :-1, 1])**2))

        for i in range(1, self.n_exits):
            self.dist[:, :] = np.minimum(self.dist, (np.sqrt((states[:, 1:, 6 * self.n_entities]-states[:, 1:, 6*i])**2+(states[:, 1:, 6 * self.n_entities+1]-states[:, 1:, 6*i +1])**2))\
               - (np.sqrt((states[:, :-1, 6 * self.n_entities]-states[:, :-1, 6*i])**2+(states[:, :-1, 6 * self.n_entities+1]-states[:, :-1, 6*i+1])**2)))

        self['movement'] = -self.dist

        self['win'][:, :] = base_rewards

        #mask = reward_shape['win'] == 0.
        #self['movement'][mask] = 0.
        #self['velocity'][mask] = 0.

        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)

        performance = np.mean(np.sum(base_rewards/self.base[self.main], axis=0))

        return self.values, performance




