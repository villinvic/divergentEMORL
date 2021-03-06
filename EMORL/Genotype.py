import numpy as np

from config.Loader import Default
from EMORL import misc
from EMORL.RL import Policy, AC
from collections import deque
from copy import deepcopy


class Genotype(Default):
    _base_keys = [
        'learning',
        'experience',
    ]

    _special = [
        'brain'
    ]

    def __init__(self, input_dim, output_dim, batch_dim=(1,1), trainable=False, is_dummy=False):
        super(Genotype, self).__init__()
        self.is_dummy = is_dummy
        self.layer_dims = []
        self.lstm_dim = 0

        if is_dummy:
            self._genes = {
                'brain' : None,
                'learning': None,
                'experience': None,
            }

        else:
            for type, dimension in self.brain_model:
                if type == 'Dense':
                    self.layer_dims.append(dimension)
                elif type == 'LSTM':
                    self.lstm_dim = dimension
                else:
                    print('Unsupported layer type... (%s)' %type)

            # get rid of the panda data
            del self.brain_model

            Brain_cls = AC if trainable else Policy

            self._genes = {
                'brain': Brain_cls(output_dim, self.layer_dims, self.lstm_dim), # brain function (must have a __call__ and perturb function) Usually an NN
                'learning': LearningParams(),
                'experience': RewardShape(),
            }

            self._genes['brain'].init_body(np.zeros((batch_dim+(input_dim,))))

    def perturb(self):
        if not self.is_dummy:
            for gene_family in self._genes.values():
                if gene_family is not None:
                    gene_family.perturb()

    def get_params(self, full_brain=False, trainable=False):
        if full_brain:
            c = {'brain': self._genes['brain']}
        elif trainable:
            c= {'brain': self._genes['brain'].get_training_params()}
        else:
            c = {'brain': self._genes['brain'].get_params()}

        c.update({key: self._genes[key].copy() for key in self._base_keys})

        return c

    def set_params(self, new_genes, trainable=False):
        if trainable:
            self._genes['brain'].set_training_params(new_genes['brain'])
        else:
            self._genes['brain'].set_params(new_genes['brain'])

        for key in self._base_keys:
            self._genes[key] = new_genes[key]



    def __repr__(self):
        return self._genes.__repr__()

    def __getitem__(self, item):
        return self._genes[item]

    def __setitem__(self, key, value):
        self._genes[key] = value

    def crossover(self, other_genotype, target_genotype):
        genes = self.get_params(trainable=True)
        target_genotype.set_params(genes, trainable=True)

        for gene_family, other_gene_family in zip(target_genotype._genes.values(), other_genotype._genes.values()):
            if gene_family is not None:
                gene_family.crossover(other_gene_family)



class EvolvingFamily:
    def __init__(self):
        self._variables = {name: EvolvingVariable(name, (domain_lower, domain_higher), self.perturb_power,
                                                 self.perturb_chance, self.reset_chance) for name, domain_lower, domain_higher
                          in self.variable_base}

        # get rid of the panda data
        del self.variable_base

    def __getitem__(self, item):
        return self._variables[item].get()

    def perturb(self):
        for variable in self._variables.values():
            variable.perturb()

    def crossover(self, other_family):
        for variable, other_variable in zip(self._variables.values(), other_family._variables.values()):
            variable.crossover(other_variable)

    def __repr__(self):
        return self._variables.__repr__()


class RewardShape(Default, EvolvingFamily):
    def __init__(self):
        super().__init__()

    def copy(self):
        new = RewardShape()
        new._variables = {}
        for k, v in self._variables.items():
            new._variables[k] = v.copy()
        return new


class LearningParams(Default, EvolvingFamily):
    def __init__(self):
        super().__init__()


    def copy(self):
        new = LearningParams()
        new._variables = {}
        for k, v in self._variables.items():
            new._variables[k] = v.copy()
        return new


class EvolvingVariable(Default):
    def __init__(self, name, domain, perturb_power=0.2, perturb_chance=0.05, reset_chance=0.1, frozen=False):
        super(EvolvingVariable, self).__init__()
        self.name = name
        if 0. in domain:
            self.domain = (0,0)
            self._current_value = 0.
        else:
            self.domain = domain
            self._current_value = misc.log_uniform(*domain)
        self.perturb_power = perturb_power

        if name == 'gamma':
            self.perturb_power = 0.0015
        self.perturb_chance = perturb_chance
        self.reset_chance = reset_chance
        self.history = deque([self._current_value], maxlen=int(self.history_max))

        self.frozen = frozen

    def perturb(self):
        if not self.frozen and np.random.random() < self.perturb_chance:
            if np.random.random() < self.reset_chance and 0. not in self.domain:
                self._current_value = misc.log_uniform(*self.domain)
            else:
                perturbation = np.random.choice([1.-self.perturb_power, 1.+self.perturb_power])
                self._current_value = np.clip(perturbation * self._current_value, *self.domain) # clip ??
            self.history.append(self._current_value)

    def crossover(self, other_variable):
        t = np.random.uniform(-(1+self.perturb_power), 1+self.perturb_power)
        self._current_value = np.clip((1 - t) * self._current_value + t * other_variable._current_value, *self.domain)
        #if np.random.random() < 0.5:
        #    self._current_value = other_variable._current_value

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return str(self._current_value)

    def get(self):
        return self._current_value

    def copy(self):
        new = EvolvingVariable(name=self.name, domain=(0,1), perturb_power=self.perturb_power, perturb_chance=self.perturb_chance, frozen=self.frozen)
        new.domain = self.domain
        new._current_value = self._current_value
        new.history = deepcopy(self.history)

        return new