import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from multiprocessing import set_start_method, get_context
import numpy as np
import fire
import logging
import tensorflow as tf
import multiprocessing

from EMORL.Individual import Individual
from EMORL.Population import Population
from EMORL.plotting import plot_stats, heatmap
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


def eval_behav(args):
    k, genotype = args
    game = Game()

    player = Individual(-1, game.state_dim, game.action_dim, [])
    player.set_arena_genes(genotype)
    print(k)

    states = np.empty((1000000, game.state_dim), dtype=np.float32)
    n_games = 15
    final_states = np.empty((n_games,), dtype=np.int32)
    points = [0,0]
    state_idx = 0
    try:
        for i in range(n_games):
            done = False
            while not done:
                states[state_idx] = game.state
                state_idx += 1
                action_id, distribution, hidden_h, hidden_c = player.policy(game.state)
                done, win = game.step(action_id)
            if win == 1:
                points[0] += 1
            else:
                points[1] += 1
            final_states[i] = state_idx - 1
            print(i)
            game.reset()
            if player.genotype['brain'].has_lstm:
                player.genotype['brain'].lstm.reset_states()
    except KeyboardInterrupt:
        pass
    stats = game.compute_stats(states[:state_idx], final_states, points)
    heatmap(game.locations(states[:state_idx][::100]), 'results/', 'location_heatmap_' + str(k), title='Location heatmap for individual ' + str(k))
    heatmap(game.punch_locations(states[:state_idx]), 'results/', 'punches_heatmap_' + str(k), title='Punches location heatmap for individual ' + str(k))

    print('-------individual', k, '-------')
    pprint(stats)



def load_and_stuff(path, pop_size, stuff='plot'):
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
    set_start_method("spawn")

    def plot():

        plot_stats(pop, '')

    def visualize_pop():
        for i, individual in enumerate(pop):
            player.set_arena_genes(individual.get_arena_genes())
            if player.genotype['brain'].has_lstm:
                player.genotype['brain'].lstm.reset_states()
            print('-------individual', i, '-------')
            pprint([individual.genotype['learning'],individual.genotype['experience']])
            pprint(['performance:', individual.performance])

            play_episode(game, player)

    def eval_pop():
        all_genes = [(i, individual.get_arena_genes()) for i, individual in enumerate(pop)]
        with get_context("spawn").Pool() as pool:
            pool.map(eval_behav, all_genes)




    {'plot': plot,
     'play': visualize_pop,
     'eval': eval_pop
     }[stuff]()


if __name__ == '__main__':
    fire.Fire(load_and_stuff)
