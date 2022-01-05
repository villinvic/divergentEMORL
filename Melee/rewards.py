import numpy as np

from Melee.game.state import GameState
from Melee.game.state import PlayerState


class Rewards:
    base = {
        'hit_dmg': PlayerState.scales[PlayerState.indexes['percent']],
        'hurt_dmg': PlayerState.scales[PlayerState.indexes['percent']],
        'kill': PlayerState.scales[PlayerState.indexes['stocks']],
        'death': PlayerState.scales[PlayerState.indexes['stocks']],
        'distance': PlayerState.scales[PlayerState.indexes['x']],
        'win': 1.,
    }

    special = np.array([
        'combo'
        #'negative_scale'
        'action_state_entropy'
    ])

    main = np.array([
        'win'
    ])


    def __init__(self, batch_size, trajectory_length):

        self.scores = {
            name : np.zeros((batch_size, trajectory_length-1), dtype=np.float32) for name, scale in self.base.items()
        }

        self.values = np.zeros((batch_size, trajectory_length-1), dtype=np.float32)

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, item):
        return self.scores[item]

    def compute(self, states, reward_shape):
        """
        states[b,t, state]
        rewards[b,t-1, rewards]

        'hit_dmg',
        'hurt_dmg',
        'kill',
        'death',
        'death_ally',
        'is_combo',
        'win'
        """

        combo_p1_multiplier = 1. + np.float32(states[:, 1:, GameState.indexes['p1_hitstun_left']] >= 1) * reward_shape['combo']

        self['hit_dmg'] =  np.maximum(states[:, 1:, GameState.indexes['p1_percent']] - states[:, :-1, GameState.indexes['p1_percent']], 0.) * combo_p1_multiplier

        self['hurt_dmg'] = -np.maximum(states[:, 1:, GameState.indexes['p0_percent']] - states[:, :-1, GameState.indexes['p0_percent']], 0.)
        self['kill'] = np.maximum(states[:, :-1, GameState.indexes['p1_stocks']] - states[:, 1:, GameState.indexes['p1_stocks']], 0.)

        self['death'] = -np.maximum(states[:, :-1, GameState.indexes['p0_stocks']] - states[:, 1:, GameState.indexes['p0_stocks']], 0.)


        self['distance'] = - np.sqrt( np.square(states[:, 1:, GameState.indexes['p1_x']] - states[:, 1:, GameState.indexes['p0_x']] )
                                    + np.square(states[:, 1:, GameState.indexes['p1_y']] - states[:, 1:, GameState.indexes['p0_y']])) \
                           + np.sqrt( np.square(states[:, :-1, GameState.indexes['p1_x']] - states[:, :-1, GameState.indexes['p0_x']])
                                    + np.square(states[:, :-1, GameState.indexes['p1_y']] - states[:, :-1, GameState.indexes['p0_y']]))

        win = np.float32(np.logical_and(
            states[:, 1:, GameState.indexes['p1_stocks']] < 1e-4,
            self['kill'] > 0))
        loss = np.float32(np.logical_and(
            states[:, 1:, GameState.indexes['p0_stocks']] < 1e-4,
            self['death'] < 0))

        self['win'] = win - loss

        p0_stocks = (1-states[:, 1:, GameState.indexes['p0_stocks']]) * (win+loss)
        p1_stocks = (1-states[:, 1:, GameState.indexes['p1_stocks']]) * (win+loss)


        # use np arrays instead of dicts...
        self.values[:, :] = np.sum([self[event]*reward_shape[event]/state_scale for event, state_scale in self.base.items()], axis=0)
        #self.values[:, :] = (1.0 - reward_shape['negative_scale']) * np.maximum(total, 0.) + total
        if np.sum(p0_stocks+p1_stocks) > 0:
            wr = np.sum(p1_stocks) / np.sum((p0_stocks+p1_stocks))

        else:
            wr = np.nan
        return self.values, wr




