import zmq
import pandas as pd
import numpy as np
import tensorflow as tf
from time import time
import datetime
import fire
import os
import pickle

from EMORLMA.Population import Population
from EMORLMA.Individual import Individual
from EMORLMA.misc import policy_similarity, MovingAverage, rbf_kernel
from EMORLMA.MOO import ND_sort
from EMORLMA.plotting import plot_perf_uniq
from Gym.BoxingMA import BoxingMA
from Gym.rewards import BoxingRewards
from config.Loader import Default
from logger.Logger import Logger


class Hub(Default, Logger):
    def __init__(self, ip='127.0.0.1', ckpt='', skip_init=False):
        super(Hub, self).__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.summary.experimental.set_step(0)

        #dummy_env = Game()
        dummy_env = BoxingMA()
        self.max_entropy = np.log(dummy_env.action_dim)

        self.running_instance_id = datetime.datetime.now().strftime("EMORL_%Y-%m-%d_%H-%M")
        self.logger.info("Hub started at" + ip)

        self.reference = Individual(-1, dummy_env.state_dim, dummy_env.action_dim, [], trainable=True)
        with open(self.reference_path, 'rb') as f:
            self.reference.set_all(pickle.load(f))
        self.reference.elo.start = 2000
        self.reference.elo.locked = True

        self.logger.info("Population Initialization started...")
        self.population = Population(self.pop_size, dummy_env.state_dim, dummy_env.action_dim)
        self.population.initialize(trainable=True, batch_dim=(self.BATCH_SIZE, self.TRAJECTORY_LENGTH))

        self.offspring_pool = Population(self.n_offspring, dummy_env.state_dim, dummy_env.action_dim)
        self.offspring_pool.initialize(trainable=True, batch_dim=(self.BATCH_SIZE, self.TRAJECTORY_LENGTH))

        # tajectories used to compute behavior distance
        self.sampled_trajectory = np.zeros((1, self.sample_size, dummy_env.state_dim), dtype=np.float32)
        self.sampled_trajectory_tmp = np.zeros_like(self.sampled_trajectory)
        self.traj_index = 0
        self.behavior_embeddings = np.zeros((self.pop_size+self.n_offspring, 1,
                                              self.sample_size, dummy_env.action_dim), dtype=np.float32)

        #self.behavior_embeddings_elite = np.zeros((self.top_k, 1,
        #                                     self.TRAJECTORY_LENGTH - 1, dummy_env.action_dim), dtype=np.float32)
        self.perf_and_uniqueness = np.zeros((2, self.pop_size+self.n_offspring, 1), dtype=np.float32)
        self.div_scores = np.zeros((self.n_offspring+self.pop_size), dtype=np.float32)

        self.rewards = BoxingRewards(self.BATCH_SIZE, self.TRAJECTORY_LENGTH)
            #Rewards( self.BATCH_SIZE, self.TRAJECTORY_LENGTH, dummy_env.area_size, dummy_env.max_see, dummy_env.view_range)

        c = zmq.Context()
        self.blob_socket = c.socket(zmq.REP)
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

        self.policy_kernel = np.empty((self.pop_size, self.pop_size), dtype=np.float32)
        # self.policy_kernel_p1 = np.empty((self.pop_size+self.n_offspring, self.top_k+self.n_offspring), dtype=np.float32)
        self.init_sampled_trajectories(dummy_env)

        if ckpt:
            self.load(ckpt)

    def init_sampled_trajectories(self, dummy_env):
        for j in range(self.sample_size):

            n_steps = 5
            for _ in range(n_steps):
                done, _ = dummy_env.step(np.random.choice(dummy_env.action_dim, (2,)))
                if done:
                    dummy_env.reset()
            self.sampled_trajectory[0, j, :] = dummy_env.state#/ dummy_env.scales

        self.sampled_trajectory_tmp[:] = self.sampled_trajectory

    def recv_training_data(self):
        received = 0
        try:
            while True:
                self.exp.append(self.exp_socket.recv_pyobj(zmq.NOBLOCK))
                received += 1
        except zmq.ZMQError:
            pass
        if len(self.exp)> self.BATCH_SIZE * 3:
            self.logger.info('exp waiting: %d ' % len(self.exp))
        self.rcved += received

    def handle_requests(self, index):
        try:
            match_result, player_ids = self.blob_socket.recv_pyobj(zmq.NOBLOCK)
            self.logger.info((match_result, player_ids))
            if match_result is not None:
                p1 = self.index_to_individual(player_ids[0])
                p2 = self.index_to_individual(player_ids[1])

                outcome = (match_result + 1 )/2.
                p1.elo.update(p1.elo(), p2.elo(), np.float32(outcome))
                p2.elo.update(p2.elo(), p1.elo(), np.float32(not outcome))

            self.blob_socket.send_pyobj(self.matchmaking())
        except zmq.ZMQError:
            pass

    def matchmaking(self):
        # random matchmaking policy
        idxs = [np.random.choice(np.arange(-1, self.pop_size+self.n_offspring), 1, replace=False), self.pop_size]
        np.random.shuffle(idxs)
        return [self.index_to_individual(idx).get_arena_genes() for idx in idxs], idxs

    def index_to_individual(self, idx):
        if idx < 0:
            return self.reference
        elif idx < self.pop_size:
            return self.population[idx]
        else:
            return self.offspring_pool[idx-self.pop_size]

    def compute_diversity(self):
        for index, i in enumerate(self.population):
                self.behavior_embeddings[self.n_offspring+index, :, :, :] = i.probabilities_for(self.sampled_trajectory)

        self.policy_kernel[:] = rbf_kernel(self.behavior_embeddings[self.n_offspring:].reshape((self.pop_size, np.prod(self.behavior_embeddings.shape[1:]))), self.l)

        div = np.linalg.det(self.policy_kernel)
        print(self.policy_kernel)
        self.population.diversity = div
        return div

    def fake_train(self, index):
        if len(self.exp) >= self.BATCH_SIZE:
            # Get experience from the queue
            trajectory = pd.DataFrame(self.exp[:self.BATCH_SIZE]).values
            self.exp = self.exp[self.BATCH_SIZE:]

            # Cook data
            states = np.float32(np.stack(trajectory[:, 0], axis=0))
            wins = np.float32(np.stack(trajectory[:, 3], axis=0)[:, :-1])
            rews, performance = self.rewards.compute(states, self.population[index].genotype['experience'], wins)
            return performance

        return None

    def train(self, index, excluded):
        if len(self.exp) >= self.BATCH_SIZE:
            # Get experience from the queue
            trajectory = pd.DataFrame(self.exp[:self.BATCH_SIZE]).values
            self.exp = self.exp[self.BATCH_SIZE:]

            # Cook data
            states = np.float32(np.stack(trajectory[:, 0], axis=0))
            actions = np.float32(np.stack(trajectory[:, 1], axis=0)[:, :-1])
            probs = np.float32(np.stack(trajectory[:, 2], axis=0)[:, :-1])
            wins = np.float32(np.stack(trajectory[:, 3], axis=0)[:, :-1])
            #rewards = np.float32(np.stack(trajectory[:, 5], axis=0)[:, :-1])
            hidden_states = np.float32(np.stack(trajectory[:, 4], axis=0))
            rews, performance = self.rewards.compute(states, self.offspring_pool[index].genotype['experience'], wins)
            #rews = self.rewards.compute(rewards, self.offspring_pool[index].genotype['experience'], wins)
            #performance = np.sum(np.mean(wins, axis=0))

            # Train
            with tf.summary.record_if(self.train_cntr % self.write_summary_freq == 0):
                self.offspring_pool[index].mean_entropy = \
                    self.offspring_pool[index].genotype['brain'].train(index, self.offspring_pool[index].parent_index, self.sampled_trajectory,
                                                                       np.delete(self.behavior_embeddings[self.n_offspring:], excluded, axis=0),
                                                                       np.delete(np.delete(self.policy_kernel, excluded, axis=0), excluded, axis=1), self.l,
                                                                       self.pop_size-1,
                                                                       self.offspring_pool[index].genotype['learning'],
                                                                       states, actions, rews, probs, hidden_states, 0)
            self.train_cntr += 1
            tf.summary.experimental.set_step(self.train_cntr)

            print('train ! Elo=', self.offspring_pool[index].elo(),
                  ', H=', self.offspring_pool[index].mean_entropy, ', alpha=', self.offspring_pool[index].genotype['learning']['entropy_cost'])

            self.sample_states(states)

            return performance

        return None

    def sample_states(self, batch):
        if self.traj_index < 1 or np.random.random() < self.sample_state_chance:
            self.sampled_trajectory_tmp[0, self.traj_index % self.sample_size, :] = batch[np.random.randint(0, self.BATCH_SIZE), np.random.randint(0, self.TRAJECTORY_LENGTH)]

            self.traj_index += 1

    def __call__(self):
        try:
            while self.population.checkpoint_index < self.max_gen:
                self.population.checkpoint_index += 1
                self.logger.info('------| Starting Generation %d |------' %self.population.checkpoint_index)
                self.logger.info('Div(P)= %.3f' % self.compute_diversity())
                self.logger.info('Making offspring...')
                self.make_offspring()
                self.logger.info('Training offspring...')
                self.train_offspring()
                self.sampled_trajectory[:] = self.sampled_trajectory_tmp
                self.logger.info('Computing uniqueness...')
                self.compute_embeddings()
                self.compute_div_scores()
                self.logger.info('Selecting...')
                self.select()
                self.save()

        except KeyboardInterrupt:
            self.save()

        self.logger.info('Hub exited.')

    def make_offspring(self):
        # select p pairs of individuals, operate crossover with low prob, mutation
        parents = np.random.choice(self.pop_size, (self.n_offspring, 2), replace=False)

        for offspring, pair in zip(self.offspring_pool, parents):
            if np.random.random() < self.crossover_rate:
                offspring.inerit_from(self.population[pair[0]])
                offspring.parent_index = pair[0]
            else:
                offspring.inerit_from(*self.population[pair])
                offspring.parent_index = np.random.choice(pair)

            if np.random.random() < self.mutation_rate:
                offspring.perturb()
            offspring.generation = self.population.checkpoint_index

    def select_k(self, excluded):
        # for each individual, select k random (or nearest ?)  individual behavior stats as a landmark for divergence
        return self.population[np.random.choice(np.delete(np.arange(self.pop_size), excluded, 0), self.k_random)]

    def train_offspring(self):
        # Train each individuals for x minutes, 1 by 1, on the trainer (can be the Hub, with GPU)
        # empty queue
        for index in range(self.n_offspring):
            self.logger.info('Training offspring n°%d...' % index)
            start_time = time()
            for _ in range(6):
                self.recv_training_data()
            del self.exp[:]
            excluded = np.random.choice(self.pop_size)
            while time() - start_time < self.train_time:
                self.handle_requests(index)
                self.recv_training_data()
                self.train(index, excluded)

    def compute_embeddings(self):
        index = 0
        for pop in [self.offspring_pool, self.population]:
            for individual in pop:
                self.behavior_embeddings[index, :, :, :] = individual.probabilities_for(self.sampled_trajectory)
                index += 1

    def compute_div_scores(self):
        self.offspring_pool[0].div_score = 1-self.population.diversity
        self.div_scores[0] = 1-self.population.diversity
        self.perf_and_uniqueness[1, -1, 0] = - np.log(self.population.diversity)

        for index in range(self.pop_size):
            tmp = np.delete(self.behavior_embeddings, 1+index, axis=0)
            self.policy_kernel[:] = rbf_kernel(tmp.reshape((self.pop_size, np.prod(tmp.shape[1:]))), self.l)
            div = np.linalg.det(self.policy_kernel)
            self.population[index].div_score = 1-div
            self.perf_and_uniqueness[1, index, 0] = - np.log(div)

    def select(self):
        index = 0
        for i, pop in enumerate([self.population, self.offspring_pool]):
            for individual in pop:
                if i == 1 and (individual.elo() < np.min(self.perf_and_uniqueness[0, :-self.n_offspring, 0])):
                    self.perf_and_uniqueness[0, index, 0] = 0
                    individual.performance = 0
                    self.perf_and_uniqueness[1, index, 0] = 0
                else:
                    self.perf_and_uniqueness[0, index, 0] = individual.elo()
                    individual.performance = individual.elo()
                index += 1
        #self.compute_novelty()

        frontiers = ND_sort(self.perf_and_uniqueness, epsilon=self.epsilon)
        selected = []
        frontier_index = 0
        while len(selected) < self.pop_size:
            if len(frontiers[frontier_index])+len(selected) > self.pop_size:
                #ranked_by_uniqueness = list(sorted(frontiers[frontier_index],
                #                                            key=lambda index: self.perf_and_uniqueness[0, index, 0]))[len(selected)-self.pop_size:]
                random = frontiers[frontier_index][len(selected)-self.pop_size:]
                selected.extend(random)
            else:
                selected.extend(frontiers[frontier_index])
            frontier_index += 1

        # update elites regarding div
        """
        for s in selected:
            if s >= self.pop_size:
                index = s-self.pop_size
                print('===============', self.div_scores[index],  self.div_scores[self.n_offspring:], '===============')
                if self.div_scores[index] > np.min(self.div_scores[self.n_offspring:]) and self.perf_and_uniqueness[0,s,0]+self.epsilon > np.min(self.perf_and_uniqueness[0,-self.top_k:,0]):
                    self.logger.info('New elite !')
                    self.elites[np.argmin(self.div_scores[self.n_offspring:])].inerit_from(self.offspring_pool[index])
        """

        # get stats of selection...
        full_path = 'checkpoints/' + self.running_instance_id + '/ckpt_' + str(
            self.population.checkpoint_index) + '/'
        plot_perf_uniq(self.perf_and_uniqueness[:, :, 0], selected, self.population, self.reference.elo(), full_path)

        print(self.perf_and_uniqueness[:, selected, 0])

        for new_index, individual_index in enumerate(sorted(selected)):
            if individual_index < self.pop_size:
                self.population[new_index].inerit_from(self.population[individual_index])
            else:
                self.population[new_index].inerit_from(self.offspring_pool[individual_index-self.pop_size])

    def compute_novelty(self):
        # UNUSED
        all_embedings = np.concatenate([self.behavior_embeddings_pop, self.behavior_embeddings], axis=0)
        all_embedings = all_embedings.reshape((all_embedings.shape[0], np.prod(all_embedings.shape[1:])))
        for individual_index in range(len(all_embedings)):
            distance = 0.
            for individual2_index in range(len(all_embedings)):
                if individual_index != individual2_index:
                    distance += 1. - policy_similarity(all_embedings[individual_index],
                                                       all_embedings[individual2_index],
                                                       self.l)
            distance /= np.float32((len(all_embedings)-1))
            self.perf_and_uniqueness[1, individual_index] = distance


    def load(self, ckpt_path):
        self.logger.info('Loading checkpoint %s ...' % ckpt_path)
        self.population.load(ckpt_path)
        #self.elites.load(ckpt_path + 'elites/')
        path_dirs = ckpt_path.split('/')
        for d in path_dirs:
            if 'EMORL' in d:
                self.running_instance_id = d

    def save(self):
        # save pop
        self.logger.info('Saving population and parameters...')
        ckpt_path = 'checkpoints/' + self.running_instance_id + '/'
        full_path = ckpt_path + 'ckpt_' + str(self.population.checkpoint_index) + '/'

        # plot scores (perf in function of kl, gene values, behavior stats...)
        self.population.save(full_path)
        #self.elites.save(full_path+'elites/')
        # plot_stats(self.population, full_path)

        _, dirs, _ = next(os.walk(ckpt_path))

        if len(dirs) > self.ckpt_keep:
            oldest = sorted(dirs, key=lambda dir: int(dir.split('ckpt_')[-1]))[0]
            _, dirs, files = next(os.walk(ckpt_path + oldest))
            for f in files:
                if '.pkl' in f or '.params' in f or '.png' in f:
                    os.remove(ckpt_path + oldest + '/' + f)
            for d in dirs:
                if d == 'elites':
                    _, _, ffiles = next(os.walk(ckpt_path + oldest + '/' + d))
                    for f in ffiles:
                        if '.pkl' in f or '.params' in f:
                            os.remove(ckpt_path + oldest + '/' + d + '/' + f)
                    os.rmdir(ckpt_path + oldest + '/' + d)
            try:
                os.rmdir(ckpt_path + oldest)
            except Exception:
                self.logger.warning("Tried to delete a non empty checkpoint directory")


if __name__ == '__main__':
    fire.Fire(Hub)
