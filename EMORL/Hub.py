import zmq
import pandas as pd
import numpy as np
import tensorflow as tf
from time import time
import datetime
import fire
import os

from EMORL.Population import Population
from EMORL.misc import kl_divergence
from EMORL.MOO import ND_sort
from EMORL.plotting import plot_stats
from Game.core import Game
from Game.rewards import Rewards
from config.Loader import Default
from logger.Logger import Logger


class Hub(Default, Logger):
    def __init__(self, ip='127.0.0.1', ckpt=''):
        super(Hub, self).__init__()
        dummy_env = Game()
        self.max_entropy = np.log(dummy_env.action_dim)

        self.running_instance_id = datetime.datetime.now().strftime("EMORL_%Y-%m-%d_%H-%M")
        self.logger.info("Hub started at" + ip)

        self.logger.info("Population Initialization started...")
        self.population = Population(self.pop_size, dummy_env.state_dim, dummy_env.action_dim)
        if ckpt:
            self.load(ckpt)
        self.offspring_pool = Population(self.n_offspring, dummy_env.state_dim, dummy_env.action_dim)
        # Init pop
        self.population.initialize(trainable=True)
        self.offspring_pool.initialize(trainable=True)

        # tajectories used to compute behavior distance
        self.sampled_trajectories = np.zeros((self.n_traj_ref, self.BATCH_SIZE, self.TRAJECTORY_LENGTH, dummy_env.state_dim), dtype=np.float32)
        self.traj_index = 0
        self.policy_distributions = np.zeros((self.pop_size+self.n_offspring, self.n_traj_ref, self.BATCH_SIZE,
                                              self.TRAJECTORY_LENGTH-1, dummy_env.action_dim), dtype=np.float32)
        self.perf_and_uniqueness = np.zeros((2, self.pop_size+self.n_offspring), dtype=np.float32)

        self.eval_queue = np.full((100,), fill_value=np.nan, dtype=np.float32)
        self.eval_index = 0

        self.rewards = Rewards( self.BATCH_SIZE, self.TRAJECTORY_LENGTH, dummy_env.area_size, dummy_env.n_cyclones, dummy_env.n_exits)

        c = zmq.Context()
        self.blob_socket = c.socket(zmq.PUB)
        self.blob_socket.bind("tcp://%s:%d" % (ip, self.PARAM_PORT))
        self.exp_socket = c.socket(zmq.PULL)
        self.exp_socket.bind("tcp://%s:%d" % (ip, self.EXP_PORT))

        self.exp = []
        self.rcved = 0
        self.train_cntr = 0
        self.write_summary_freq = 3

        tf.summary.experimental.set_step(0)
        log_dir = 'logs/train' + self.running_instance_id
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()

    def recv_training_data(self):
        received = 0
        try:
            while True:
                self.exp.append(self.exp_socket.recv_pyobj(zmq.NOBLOCK))
                received += 1
        except zmq.ZMQError:
            pass
        self.rcved += received

    def pub_params(self, index):
        try:
            # self.param_socket.send_string(str(individual.id), zmq.SNDMORE)
            # self.param_socket.send_pyobj(individual.get_arena_genes(), flags=zmq.NOBLOCK)
            self.blob_socket.send_pyobj(self.offspring_pool[index].get_arena_genes())
        except zmq.ZMQError:
            pass

    def train(self, index):
        if len(self.exp) >= self.BATCH_SIZE:
            # Get experience from the queue
            trajectory = pd.DataFrame(self.exp[:self.BATCH_SIZE]).values
            self.exp = self.exp[self.BATCH_SIZE:]

            # Cook data
            states = np.float32(np.stack(trajectory[:, 0], axis=0))
            actions = np.float32(np.stack(trajectory[:, 1], axis=0)[:, :-1])
            probs = np.float32(np.stack(trajectory[:, 2], axis=0)[:, :-1])
            wins = np.float32(np.stack(trajectory[:, 3], axis=0)[:, :-1])
            rews, performance = self.rewards.compute(states, self.offspring_pool[index].genotype['experience'], wins)
            # Train
            with tf.summary.record_if(self.train_cntr % self.write_summary_freq == 0):

                self.offspring_pool[index].mean_entropy = \
                    self.offspring_pool[index].genotype['brain'].train(str(index), self.offspring_pool[index].genotype['learning'],states, actions, rews, probs, 0)
            self.train_cntr += 1
            tf.summary.experimental.set_step(self.train_cntr)

            print('train !', self.offspring_pool[index].mean_entropy)

            self.sample_trajectory(states)



            return performance

        return None

    def sample_trajectory(self, sample):
        if self.traj_index < self.n_traj_ref or np.random.random() < self.save_traj_batch_chance:
            self.sampled_trajectories[self.traj_index%self.n_traj_ref, :, :, :] = sample
            self.traj_index += 1

    def __call__(self):
        try:
            while True:
                self.logger.info('------| Starting Generation %d |------' %self.population.checkpoint_index)
                self.logger.info('Making offspring...')
                self.make_offspring()
                self.logger.info('Training offspring...')
                self.train_offspring()
                self.logger.info('Computing uniqueness...')
                self.compute_uniqueness()
                self.logger.info('Selecting...')
                self.select()
                self.population.checkpoint_index += 1
                self.save()
        except KeyboardInterrupt:
            pass

        self.save()
        self.logger.info('Hub exited.')

    def make_offspring(self):
        # select p pairs of individuals, operate crossover with low prob, mutation
        parents_pairs = np.random.choice(self.pop_size, (self.n_offspring, 2), replace=True)

        for offspring, parents in zip(self.offspring_pool, parents_pairs):
            if np.random.random() < self.crossover_rate:
                offspring.inerit_from(*self.population[parents])
            else:
                offspring.inerit_from(self.population[parents[0]])

            if np.random.random() < self.mutation_rate:
                offspring.perturb()
            offspring.generation = self.population.checkpoint_index + 1

    def select_k(self):
        # for each individual, select k random (or nearest ?)  individual behavior stats as a landmark for divergence
        return self.population[np.random.choice(self.pop_size, self.k_random)]

    def reset_eval_queue(self):
        self.eval_queue[:] = np.nan
        self.eval_index = 0

    def train_offspring(self):
        # Train each individuals for x minutes, 1 by 1, on the trainer (can be the Hub, with GPU)
        for index in range(self.n_offspring):
            self.pub_params(index)
            last_pub_time = time()
            self.reset_eval_queue()
            start_time = time()
            while time() - start_time < self.train_time:
                self.recv_training_data()
                perf = self.train(index)
                if perf is not None:
                    self.eval_queue[self.eval_index % len(self.eval_queue)] = perf
                    self.eval_index = 0

                current = time()
                if current - last_pub_time > 10:
                    print('pub', index)
                    self.pub_params(index)
                    last_pub_time = current

            # use recent training eval for selection
            self.offspring_pool[index].performance = np.nanmean(self.eval_queue)

    def compute_uniqueness(self):
        index = 0
        for pop in [self.population, self.offspring_pool]:
            for individual in pop:
                for batch_index, batch in enumerate(self.sampled_trajectories):
                    if individual.mean_entropy < self.min_entropy_ratio * self.max_entropy:
                        self.policy_distributions[index, batch_index, :, :, :] = 0.
                    else:
                        self.policy_distributions[index, batch_index, :, :, :] = individual.probabilities_for(batch[:, :-1])
                index += 1

        for individual_index in range(self.pop_size+self.n_offspring):
            distance = 0
            for individual2_index in range(self.pop_size+self.n_offspring):
                if individual_index != individual2_index:
                    distance += kl_divergence(self.policy_distributions[individual_index],
                                               self.policy_distributions[individual2_index])
            distance /= (self.pop_size+self.n_offspring-1)
            self.perf_and_uniqueness[1, individual_index] = distance

    def select(self):
        index = 0
        for pop in [self.population, self.offspring_pool]:
            for individual in pop:
                self.perf_and_uniqueness[0, index] = individual.performance
                index += 1

        frontiers = ND_sort(self.perf_and_uniqueness)
        selected = []
        frontier_index = 0
        while len(selected) < self.pop_size:
            if len(frontiers[frontier_index]) > self.pop_size - len(selected):
                ranked_by_uniqueness = np.array(list(sorted(frontiers[frontier_index],
                                                            key=lambda index: self.perf_and_uniqueness[1, index])))[self.pop_size - len(selected):]
                selected.extend(ranked_by_uniqueness)
            else:
                selected.extend(frontiers[frontier_index])

        for new_index, individual_index in enumerate(selected):
            if individual_index < self.pop_size:
                self.population[new_index].inerit_from(self.population[individual_index])
            else:
                self.population[new_index].inerit_from(self.offspring_pool[individual_index-self.pop_size])

        # get stats of selection...
        full_path = 'checkpoints/' + self.running_instance_id + '/ckpt_' + str(self.population.checkpoint_index) + '/'
        plot_stats(self.perf_and_uniqueness, selected, self.population, full_path)


    def load(self, ckpt_path):
        self.logger.info('Loading checkpoint %s ...' % ckpt_path)
        self.population.load(ckpt_path)
        path_dirs = ckpt_path.split('/')
        for d in path_dirs:
            if 'EMORL' in d:
                self.running_instance_id = d

    def save(self):
        # save pop
        self.logger.info('Saving population and parameters...')
        ckpt_path = 'checkpoints/' + self.running_instance_id + '/'
        full_path = ckpt_path + 'ckpt_' + str(self.population.checkpoint_index) + '/'
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except OSError as exc:
                print(exc)

        # plot scores (perf in function of kl, gene values, behavior stats...)
        self.population.save(full_path)

        _, dirs, _ = next(os.walk(ckpt_path))
        if len(dirs) > self.ckpt_keep:
            oldest = sorted(dirs, key=lambda dir: int(dir.split('ckpt_')[-1]))[0]
            _, _, files = next(os.walk(ckpt_path + oldest))
            for f in files:
                if '.pkl' in f or '.params' in f or '.pdf' in f:
                    os.remove(ckpt_path + oldest + '/' + f)
            try:
                os.rmdir(ckpt_path + oldest)
            except Exception:
                self.logger.warning("Tried to delete a non empty checkpoint directory")


if __name__ == '__main__':
    fire.Fire(Hub)
