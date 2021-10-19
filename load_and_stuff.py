from EMORL.Population import Population
from EMORL.plotting import plot_stats
from Game.core import Game
import fire

def load_and_stuff(path, pop_size):
    dummy = Game()
    pop = Population(pop_size, dummy.state_dim, dummy.action_dim)
    pop.initialize(trainable=True, batch_dim=(256,80))

    pop.load(path)
    plot_stats(pop, '')

if __name__ == '__main__':
    fire.Fire(load_and_stuff)