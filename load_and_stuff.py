import os

import fire
import tensorflow as tf

from EMORL.Individual import Individual
from EMORL.Population import Population
from EMORL.plotting import plot_stats
# from Game.core import Game
from pprint import pprint
from Gym.Boxing import Boxing as Game


def play_episode(game, player):
    done = False
    try:
        while not done:
            game.render()
            action_id, distribution, hidden_h, hidden_c = player.policy(game.state)
            done, win = game.step(action_id)
    except KeyboardInterrupt:
        pass
    game.reset()


def load_and_stuff(path, pop_size, stuff='plot'):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.summary.experimental.set_step(0)
    game = Game()

    pop = Population(pop_size, game.state_dim, game.action_dim)
    pop.initialize(trainable=True, batch_dim=(128, 80))
    pop.load(path)
    player = Individual(-1, game.state_dim, game.action_dim, [])

    def plot():

        plot_stats(pop, '')

    def visualize_pop():
        for i, individual in enumerate(pop):
            player.set_arena_genes(individual.get_arena_genes())
            player.genotype['brain'].lstm.reset_states()
            print('-------individual', i, '-------')
            pprint([individual.genotype['learning'],individual.genotype['experience']])

            play_episode(game, player)

    {'plot': plot,
     'play': visualize_pop
     }[stuff]()


if __name__ == '__main__':
    fire.Fire(load_and_stuff)
