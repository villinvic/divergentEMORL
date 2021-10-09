import numpy as np
import tensorflow as tf

from EMORL.Behavior import Behavior
from EMORL.Genotype import Genotype



class Individual:
    def __init__(self, ID, input_dim, output_dim, behavior_categories, trainable=False):
        self.id = ID
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.behavior = Behavior(behavior_categories)
        self.genotype = Genotype(input_dim, output_dim, trainable=trainable)
        self.mean_entropy = np.inf

        self.performance = 0
        self.generation = 0



    def get_arena_genes(self):
        return {
            'brain': self.genotype['brain'].get_params()
        }

    def set_arena_genes(self, arena_genes):
        self.genotype['brain'].set_params(arena_genes['brain'])

    def get_genes(self):
        return self.genotype.get_params()

    def set_genes(self, new_genes):
        self.genotype.set_params(new_genes)

    def get_all(self, trainable=True):
        return dict(
            id=self.id,
            genotype=self.genotype.get_params(trainable=trainable),
        )

    def set_all(self, params, trainable=True):
        self.genotype.set_params(params['genotype'], trainable)

    def policy(self, observation):
        if self.genotype['brain'] is None:
            # random policy
            return np.random.randint(0, self.input_dim), 0
        else:
            return self.genotype['brain'](observation)

    def inerit_from(self, *other_individuals):
        if len(other_individuals) == 1:
            self.genotype.set_params(other_individuals[0].genotype.get_params(trainable=True), trainable=True)
            self.mean_entropy = other_individuals[0].mean_entropy
            self.performance = other_individuals[0].performance
            self.generation = other_individuals[0].generation
        elif len(other_individuals) == 2:
            self.genotype.set_params(other_individuals[0].genotype.crossover(other_individuals[1].genotype),
                                     trainable=True)
            self.mean_entropy = np.inf
            self.performance = 0

    @tf.function
    def probabilities_for(self, states):
        with tf.device("/gpu:{}".format(0)):
            p = self.genotype['brain'].policy.get_probs(states)
        return p



    def perturb(self):
        self.genotype.perturb()



