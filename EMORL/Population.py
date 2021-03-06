import numpy as np
import pickle
import json
import os

from EMORL.Individual import Individual


class Population:
    def __init__(self, size, input_dim, output_dim):

        self.individuals = np.empty((size,), dtype=Individual)
        self.size = size
        self.checkpoint_index = 0
        self.n = 0
        self.diversity = 0
        self.dims = (input_dim, output_dim)

        self.to_serializable_v = np.vectorize(lambda individual: individual.get_all())
        self.read_pickled_v = np.vectorize(lambda individual, x: individual.set_all(x))

        self.stats = {
            'entropy': [],
            'performance': [],
            'diversity': [],
            'hyperparameter':
                {
                    'learning': [],
                    'experience': [],
                }
            ,
        }

    def register_generation(self):
        entropy = []
        performance = []
        learning = []
        experience = []

        for individual in self:
            entropy.append(individual.mean_entropy)
            performance.append(individual.performance)
            learning.append(individual.genotype['learning'].copy())
            experience.append(individual.genotype['experience'].copy())

        self.stats['entropy'].append(entropy)
        self.stats['performance'].append(performance)
        self.stats['hyperparameter']['learning'].append(learning)
        self.stats['hyperparameter']['experience'].append(experience)
        self.stats['diversity'].append(self.diversity)


    def __getitem__(self, item):
        return self.individuals[item]

    def __setitem__(self, key, value):
        self.individuals[key] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.size:
            individual = self.individuals[self.n]
            self.n += 1
            return individual
        else:
            raise StopIteration

    def initialize(self, trainable=False, batch_dim=(1,1)):
        for ID in range(self.size):
            self.individuals[ID] = Individual(ID, *self.dims, [], batch_dim=batch_dim, trainable=trainable)

    def __repr__(self):
        return self.individuals.__repr__()

    def to_serializable(self):
        return self.to_serializable_v(self.individuals)

    def read_pickled(self, params):
        self.read_pickled_v(self.individuals[:self.size], params)

    def save(self, path):
        self.register_generation()
        for index, individual in enumerate(self):
            with open(path + str(index) + '.pkl',
                      'wb+') as f:
                pickle.dump(individual.get_all(), f)
        with open(path + 'population.params', 'wb+') as param_file:
            pickle.dump({
            "size": int(self.size),
            "checkpoint_index": int(self.checkpoint_index),
            "stats": self.stats,
        }, param_file)

    def load(self, path):
        if path[-1] != '/':
            path += '/'
        _, _, ckpts = next(os.walk(path))
        loaded = []
        for ckpt in ckpts:
            if '.pkl' in ckpt:
                try:
                    with open(path + ckpt, 'rb') as f:
                        data = pickle.load(f)
                        if data['id'] in loaded:
                            print('Duplicated id ?')
                        else:
                            self[data['id']].set_all(data)
                            loaded.append(data['id'])
                except Exception as e:
                    print(e)
        try:
            with open(path + 'population.params',
                      'rb') as param_file:
                params = pickle.load(param_file)
            for param_name, value in params.items():
                setattr(self, param_name, value)
        except Exception as e:
            print(e)


