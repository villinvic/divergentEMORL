import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from multiprocessing import set_start_method, get_context
import numpy as np
import fire
import logging
import tensorflow as tf
import multiprocessing
import pickle
import time

from EMORL.Individual import Individual
from EMORL.Population import Population
from EMORL.plotting import plot_stats, heatmap
# from Game.core import Game
from pprint import pprint
#from Gym.Boxing import Boxing as Game
#from Gym.Kfm import Kfm as Game
from Gym.Tennis import Tennis as Game
from Gym.BoxingMA import BoxingMA as GameMA
from EMORL.Worker import MeleeWorker
from Melee.game.console import Console


def play_episode(game, player, opp_genes):
    if opp_genes:
        opp = Individual(-1, game.state_dim, game.action_dim, [])
        opp.set_arena_genes(opp_genes)
    done = False
    idx = 0
    try:
        while not done:
            idx += 1
            game.render()
            action_id, distribution, hidden_h, hidden_c = player.policy(game.state)
            if isinstance(game, GameMA):
                action_id_opp, _, _, _ = opp.policy(game.opp_state)
                action_id = [action_id, action_id_opp]
            done, win = game.step(action_id)
    except KeyboardInterrupt:
        pass
    game.reset()

def play_melee(worker, player, opp_genes=None):
    worker.player.set_arena_genes(player.get_arena_genes())
    worker.play_game()


def eval_behav(args):
    k, genotype, ma = args
    if ma:
        game = GameMA()
        opp = Individual(-1, game.state_dim, game.action_dim, [])
        opp.set_arena_genes(ma)
    else:
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
                if ma:
                    action_id_opp, _, _, _ = opp.policy(game.opp_state)
                    action_id = [action_id, action_id_opp]
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



def load_and_stuff(path, pop_size, stuff='plot', ma=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.summary.experimental.set_step(0)
    if ma:
        game = GameMA()
    else:
        #game = Game()
        game = Console(-1, False)

    pop = Population(pop_size, game.state_dim, game.action_dim)
    pop.initialize(trainable=True, batch_dim=(128, 80))
    pop.load(path)
    player = Individual(-1, game.state_dim, game.action_dim, [])
    if ma:
        opp = Individual(-1, game.state_dim, game.action_dim, [], trainable=True)
        with open('checkpoints/sample_ma/aggressive_2000.pkl', 'rb') as f:
            opp.set_all(pickle.load(f))


        opp_genes = opp.get_arena_genes()
    else:
        opp_genes = None
    set_start_method("spawn")

    def plot():

        plot_stats(pop, '')

    def visualize_pop():
        worker = MeleeWorker(0, True)
        for i, individual in enumerate(pop):
            player.set_arena_genes(individual.get_arena_genes())
            if player.genotype['brain'].has_lstm:
                player.genotype['brain'].lstm.reset_states()
            print('-------individual', individual.id, '-------')
            pprint([individual.genotype['learning'],individual.genotype['experience']])
            pprint(['performance:', individual.performance])


            #play_episode(game, player, opp_genes=opp_genes if ma else None)
            play_melee(worker, player)

    def eval_pop():

        all_genes = [(i, individual.get_arena_genes(), opp_genes) for i, individual in enumerate(pop)]
        with get_context("spawn").Pool() as pool:
            pool.map(eval_behav, all_genes)




    {'plot': plot,
     'play': visualize_pop,
     'eval': eval_pop
     }[stuff]()


if __name__ == '__main__':
    fire.Fire(load_and_stuff)
