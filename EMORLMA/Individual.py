import numpy as np

from EMORLMA.Behavior import Behavior
from EMORLMA.Genotype import Genotype
from EMORLMA.Elo import Elo



class Individual:
    def __init__(self, ID, input_dim, output_dim, behavior_categories, batch_dim=(1,1), trainable=False):
        self.id = ID
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.behavior = Behavior(behavior_categories)
        self.genotype = Genotype(input_dim, output_dim, batch_dim, trainable=trainable)
        self.mean_entropy = np.inf

        self.performance = -np.inf
        self.elo = Elo()
        self.div_score = 0
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
            performance=self.performance,
            elo=self.elo(),
            genotype=self.genotype.get_params(trainable=trainable),
        )

    def set_all(self, params, trainable=True):
        self.genotype.set_params(params['genotype'], trainable)
        self.performance = params['performance']
        self.elo.start = params['win_loss']

    def policy(self, observation):
        if self.genotype['brain'] is None:
            # random policy
            return np.random.randint(0, self.input_dim), 0
        else:
            return self.genotype['brain'](observation)

    def inerit_from(self, *other_individuals):
        self.elo = Elo()
        if len(other_individuals) == 1:
            self.genotype.set_params(other_individuals[0].genotype.get_params(trainable=True), trainable=True)
            self.mean_entropy = other_individuals[0].mean_entropy
            self.performance = other_individuals[0].performance
            self.generation = other_individuals[0].generation
            self.div_score = other_individuals[0].div_score
            self.elo.start = other_individuals[0].elo()
        elif len(other_individuals) == 2:
            other_individuals[0].genotype.crossover(other_individuals[1].genotype, target_genotype=self.genotype)
            self.mean_entropy = (other_individuals[0].mean_entropy + other_individuals[1].mean_entropy) * 0.5
            self.performance = (other_individuals[0].performance + other_individuals[1].performance) * 0.5
            self.generation = other_individuals[0].mean_entropy
            self.div_score = (other_individuals[0].div_score + other_individuals[1].div_score) * 0.5
            self.elo.start = (other_individuals[0].elo()+other_individuals[1].elo()) * 0.5


    def probabilities_for(self, states):
        return self.genotype['brain'].get_distribution(states)

    def perturb(self):
        self.genotype.perturb()



