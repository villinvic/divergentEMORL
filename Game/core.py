import numpy as np
from EMORL.misc import uniform_with_hole, log_uniform
from Rendering.pygame import PyGameEngine
from config.Loader import Default
from pprint import pprint

np.set_printoptions(suppress=True)





class Cyclone:
    def __init__(self, pos=(0.,0.), ray=1., deadly_fraction=0.1, nature=1, power=0.1):
        self.pos = pos
        self.ray = ray
        self.deadly_ray = deadly_fraction * ray
        self.nature = nature
        self.power = power

    def has(self, pos):
        return (self.pos[0] - pos[0])** 2 + (self.pos[1] - pos[1])** 2 < self.ray ** 2

    def kills(self, pos):
        return (self.pos[0] - pos[0]) ** 2 + (self.pos[1] - pos[1]) ** 2 < self.deadly_ray ** 2


class Game(Default):
    cyclone_sizes = np.array([20, 25., 30., 45., 55.])
    cyclone_probs = np.array([0.1, 0.15, 0.35 , 0.35, 0.05])

    directions = np.array([(0., 0.), (-1., 0.), (-0.707, 0.707), (0., 1.), (0.707, 0.707), (1., 0.),
                           (0.707, -0.707), (0., -1.), (-0.707, -0.707)])

    def __init__(self, render=False, human=False):
        self.human = human
        super(Game, self).__init__()
        self.state = np.zeros((self.max_see*6 + 4*2 + 1,), dtype=np.float32)

        self.indexes = {
            'player_x': self.max_see * 6,
            'player_y': self.max_see * 6 + 1,
            'inertia_x': self.max_see * 6 + 2,
            'inertia_y': self.max_see * 6 + 3,
    }
        self.cyclones = np.array( [Cyclone(pos=uniform_with_hole(high=self.area_size),
                                          ray=np.random.choice(Game.cyclone_sizes, p=Game.cyclone_probs),
                                          deadly_fraction=np.random.uniform(self.min_deadly_fraction, self.max_deadly_fraction),
                                          nature=1,
                                          power=0) for _ in range(self.n_exits)] +
            [Cyclone(pos=uniform_with_hole(high=self.area_size),
                                          ray=np.random.choice(Game.cyclone_sizes, p=Game.cyclone_probs),
                                          deadly_fraction=np.random.uniform(self.min_deadly_fraction, self.max_deadly_fraction),
                                          nature=-1,
                                          power=log_uniform(self.min_power, self.max_power)) for _ in range(self.n_cyclones)])
        self.engine = PyGameEngine(self.area_size, self.cyclones, self.human) if render else None

        # (x_acc, y_acc, speed/slow)

        self.scales = []
        for _ in range(self.max_see):
            self.scales += [self.area_size,  self.area_size, Game.cyclone_sizes[-1], Game.cyclone_sizes[-1]*self.max_deadly_fraction, 1., self.max_power]

        for _ in range(2):
            self.scales += [self.area_size, self.area_size, self.max_inertia, self.max_inertia]
        self.scales += [self.max_steps]

        self.scales = np.array(self.scales, dtype=np.float32)

        self.action_dim = len(self.directions)
        self.state_dim = len(self.state)

    def is_out(self):
        return self.state[self.indexes['player_x']] ** 2 + self.state[self.indexes['player_y']] ** 2 >= self.area_size ** 2

    def is_timeout(self):
        return self.state[-1] > self.max_steps

    def sees(self, c):
        return (self.state[self.indexes['player_x']]-c.pos[0]) ** 2 + (self.state[self.indexes['player_y']]-c.pos[1]) ** 2 \
               <= self.view_range ** 2

    def step(self, action_id):
        angle = self.directions[action_id]
        for direction, index in enumerate(['inertia_x', 'inertia_y']):
            if self.state[self.indexes[index]] == 0:
                self.state[self.indexes[index]] = 0.1 * angle[direction]
            else:
                self.state[self.indexes[index]] = np.clip(self.state[self.indexes[index]] + angle[direction] * 0.25, -self.max_inertia, self.max_inertia)


        self.state[self.max_see*6+4: -1] = self.state[self.max_see*6: -5]

        self.state[self.indexes['player_x']] += self.state[self.indexes['inertia_x']] * 1.25
        self.state[self.indexes['player_y']] += self.state[self.indexes['inertia_y']] * 1.25

        if self.is_out():
            return True, -1
        if self.is_timeout():
            return True, -1
        # test for cyclones
        seen = 0
        self.state[:6 * self.max_see] = 0.
        for i, c in enumerate(self.cyclones):
            if c.kills((self.state[self.indexes['player_x']], self.state[self.indexes['player_y']])):
                return True, c.nature
            if c.has((self.state[self.indexes['player_x']], self.state[self.indexes['player_y']])):

                dx = self.state[self.indexes['player_x']] - c.pos[0]
                dy = self.state[self.indexes['player_y']] - c.pos[1]
                if abs(dx) + abs(dy) > 0:
                    dx /= np.sqrt(dx**2 + dy**2)
                    dy /= np.sqrt(dx**2 + dy**2)

                    self.state[self.indexes['inertia_x']] = np.clip(self.state[self.indexes['inertia_x']] +
                                                                    dx * c.nature * c.power, -self.max_inertia,
                                                                    self.max_inertia)

                    self.state[self.indexes['inertia_y']] = np.clip(self.state[self.indexes['inertia_y']]+ dy *
                                                                    c.nature * c.power, -self.max_inertia,
                                                                    self.max_inertia)
            if seen < self.max_see and self.sees(c):
                self.state[seen * 6] = c.pos[0]
                self.state[seen * 6 + 1] = c.pos[1]
                self.state[seen * 6 + 2] = c.ray
                self.state[seen * 6 + 3] = c.deadly_ray
                self.state[seen * 6 + 4] = c.nature
                self.state[seen * 6 + 5] = c.power
                seen += 1

        # return state
        self.state[self.indexes['inertia_x']] *= self.friction
        self.state[self.indexes['inertia_y']] *= self.friction
        self.state[-1] += 1

        #pprint(self.state)

        return False, 0

    def reset(self):
        self.close()
        self.__init__(render=self.engine is not None, human=self.human)

    def render(self):
        if self.engine is None:
            print(self.state[self.n_cyclones*6:])
        else:
            return self.engine.update(self.state[self.indexes['player_x']], self.state[self.indexes['player_y']])

    def close(self):
        if self.engine is not None:
            self.engine.exit()


if __name__ == '__main__':

    g = Game(render=True, human=True)
    done = False
    win = None
    action = 0
    n_turns = 0
    while True:
        while not done:
            done, win = g.step(action)
            action = g.render()
            n_turns += 1
        g.reset()

        done = False



