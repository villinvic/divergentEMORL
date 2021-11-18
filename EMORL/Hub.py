import zmq
import pandas as pd
import numpy as np
import tensorflow as tf
from time import time
import datetime
import fire
import os

from EMORL.Population import Population
from EMORL.misc import policy_similarity, normalize, MovingAverage
from EMORL.MOO import ND_sort
from EMORL.plotting import plot_perf_uniq
from Gym.Boxing import Boxing
from Gym.rewards import BoxingRewards
from config.Loader import Default
from logger.Logger import Logger


class Hub(Default, Logger):
    def __init__(self, ip='127.0.0.1', ckpt=''):
        super(Hub, self).__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.summary.experimental.set_step(0)

        #dummy_env = Game()
        dummy_env = Boxing()
        self.max_entropy = np.log(dummy_env.action_dim)

        self.running_instance_id = datetime.datetime.now().strftime("EMORL_%Y-%m-%d_%H-%M")
        self.logger.info("Hub started at" + ip)

        self.logger.info("Population Initialization started...")
        self.population = Population(self.pop_size, dummy_env.state_dim, dummy_env.action_dim)
        self.population.initialize(trainable=True, batch_dim=(self.BATCH_SIZE, self.TRAJECTORY_LENGTH))

        self.offspring_pool = Population(self.n_offspring, dummy_env.state_dim, dummy_env.action_dim)
        self.offspring_pool.initialize(trainable=True, batch_dim=(self.BATCH_SIZE, self.TRAJECTORY_LENGTH))

        self.elites = Population(self.top_k, dummy_env.state_dim, dummy_env.action_dim)
        self.elites.initialize(trainable=True, batch_dim=(self.BATCH_SIZE, self.TRAJECTORY_LENGTH))
        for i, e in enumerate(self.elites):
            e.inerit_from(self.population[i])


        # tajectories used to compute behavior distance
        self.sampled_trajectory = np.zeros((1, self.TRAJECTORY_LENGTH, dummy_env.state_dim), dtype=np.float32)
        self.sampled_trajectory_tmp = np.zeros_like(self.sampled_trajectory)
        self.traj_index = 0
        self.behavior_embeddings = np.zeros((self.top_k+self.n_offspring, 1,
                                              self.TRAJECTORY_LENGTH-1, dummy_env.action_dim), dtype=np.float32)
        #self.behavior_embeddings_elite = np.zeros((self.top_k, 1,
        #                                     self.TRAJECTORY_LENGTH - 1, dummy_env.action_dim), dtype=np.float32)
        self.perf_and_uniqueness = np.zeros((2, self.pop_size+self.n_offspring+self.top_k, 1), dtype=np.float32)

        self.eval_queue = MovingAverage(self.moving_avg_size)

        self.rewards = BoxingRewards(self.BATCH_SIZE, self.TRAJECTORY_LENGTH)
            #Rewards( self.BATCH_SIZE, self.TRAJECTORY_LENGTH, dummy_env.area_size, dummy_env.max_see, dummy_env.view_range)

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


        #self.top_k_index = np.arange(self.top_k)
        self.policy_kernel = np.empty((self.top_k, self.top_k), dtype=np.float32)
        self.policy_kernel_p1 = np.empty((self.top_k+self.n_offspring, self.top_k+self.n_offspring), dtype=np.float32)
        self.init_sampled_trajectories(dummy_env)

        if ckpt:
            self.load(ckpt)
        else:
            self.init_eval()


    def init_sampled_trajectories(self, dummy_env):
        for j in range(self.TRAJECTORY_LENGTH):

            n_steps = 1
            for _ in range(n_steps):
                done, _ = dummy_env.step(np.random.choice(dummy_env.action_dim))
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

    def pub_params(self, index, init=False):
        if init:
            pop = self.population
        else:
            pop = self.offspring_pool
        try:
            # self.param_socket.send_string(str(individual.id), zmq.SNDMORE)
            # self.param_socket.send_pyobj(individual.get_arena_genes(), flags=zmq.NOBLOCK)
            self.blob_socket.send_pyobj(pop[index].get_arena_genes())
        except zmq.ZMQError:
            pass

    def compute_diversity(self):
        for index, e in enumerate(self.elites):
                self.behavior_embeddings[index, :, :, :] = e.probabilities_for(self.sampled_trajectory[:, :-1])
        self.behavior_embeddings[:-1] = normalize(self.behavior_embeddings[:-1])

        for i in range(self.top_k):
            for j in range(self.top_k):
                if i==j:
                    self.policy_kernel[i, j] = 1.
                elif j > i:
                    self.policy_kernel[i, j] = policy_similarity(self.behavior_embeddings[i],
                                                                 self.behavior_embeddings[j],
                                                                 l=self.similarity_l)
                else:
                    self.policy_kernel[i, j] = self.policy_kernel[j, i]

        div = np.linalg.det(self.policy_kernel)
        print(self.policy_kernel)
        self.population.diversity = div
        return div

    def init_eval(self):
        for index in range(self.pop_size):
            self.logger.info('Init offspring n°%d...' % index)
            self.pub_params(index, init=True)
            last_pub_time = time()
            start_time = time()
            for _ in range(6):
                self.recv_training_data()
            del self.exp[:]
            while time() - start_time < self.init_time:
                self.recv_training_data()
                perf = self.fake_train(index)
                if perf is not None:
                    self.eval_queue.push(perf)
                # if not improving or too low entropy, drop training
                current = time()
                if current - last_pub_time > 5:
                    self.pub_params(index, init=True)
                    last_pub_time = current

            # use recent training eval for selection
            self.population[index].performance = self.eval_queue()

            self.eval_queue.reset()

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
            #rewards = np.float32(np.stack(trajectory[:, 5], axis=0)[:, :-1])
            hidden_states = np.float32(np.stack(trajectory[:, 4], axis=0))
            rews, performance = self.rewards.compute(states, self.offspring_pool[index].genotype['experience'], wins)
            #rews = self.rewards.compute(rewards, self.offspring_pool[index].genotype['experience'], wins)
            #performance = np.sum(np.mean(wins, axis=0))

            # Train
            with tf.summary.record_if(self.train_cntr % self.write_summary_freq == 0):
                self.offspring_pool[index].mean_entropy = \
                    self.offspring_pool[index].genotype['brain'].train(index, self.offspring_pool[index].parent_index, self.sampled_trajectory,
                                                                       self.behavior_embeddings,
                                                                       self.policy_kernel, self.similarity_l,
                                                                       self.top_k,
                                                                       self.offspring_pool[index].genotype['learning'],
                                                                       states, actions, rews, probs, hidden_states, 0)
            self.train_cntr += 1
            tf.summary.experimental.set_step(self.train_cntr)

            print('train ! R=', self.eval_queue(), ', trend=', self.eval_queue.trend_count,
                  ', H=', self.offspring_pool[index].mean_entropy, ', alpha=', self.offspring_pool[index].genotype['learning']['entropy_cost'])

            self.sample_states(states)

            return performance

        return None

    def sample_states(self, batch):
        if self.traj_index < 1 or np.random.random() < self.save_traj_batch_chance:
            self.sampled_trajectory_tmp[0, :, :] = batch[np.random.randint(0, self.BATCH_SIZE)]

            self.traj_index += 1

    def __call__(self):
        try:
            while True:
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
            pass

        self.save()
        self.logger.info('Hub exited.')

    def make_offspring(self):
        # select p pairs of individuals, operate crossover with low prob, mutation
        parents_pairs = np.random.choice(self.pop_size, (self.n_offspring, 2, 1), replace=True)
        best_parents_pairs = np.empty((self.n_offspring, 2), dtype=np.int32)
        for i, (p1, p2) in enumerate(parents_pairs):
            best_parents_pairs[i, :] = sorted(p1, key= lambda index: self.population[index].performance)[-1] , \
                                       sorted(p2, key= lambda index: self.population[index].performance)[-1]

        for offspring, parents in zip(self.offspring_pool, best_parents_pairs):
            if np.random.random() < self.crossover_rate:
                offspring.inerit_from(*self.population[parents])
                offspring.parent_index = parents[0]
            else:
                offspring.inerit_from(self.population[parents[0]])
                offspring.parent_index = np.random.choice(parents)


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
            self.pub_params(index)
            last_pub_time = time()
            start_time = time()
            for _ in range(6):
                self.recv_training_data()
            del self.exp[:]
            while time() - start_time < self.train_time:
                self.recv_training_data()
                perf = self.train(index)
                if perf is not None:
                    self.eval_queue.push(perf)
                # if not improving or too low entropy, drop training
                if self.eval_queue.trend_count < -self.bad_trend_maxcount or self.offspring_pool[index].mean_entropy \
                        < self.max_entropy * self.critical_entropy_ratio:
                    self.logger.info('Dropped training !')
                    break
                current = time()
                if current - last_pub_time > 5:
                    self.pub_params(index)
                    last_pub_time = current

            # use recent training eval for selection
            self.offspring_pool[index].performance = self.eval_queue()

            self.eval_queue.reset()

    def compute_embeddings(self):
        index = 0
        """for pop in [self.population, self.offspring_pool]:
            for individual in pop:
                if individual.mean_entropy < self.min_entropy_ratio * self.max_entropy:
                    individual.performance = -np.inf
                    self.behavior_embeddings[index, :] = 0.
                    failed_indexes.append(index)
                else:
                    self.behavior_embeddings[index, :] = individual.probabilities_for(self.sampled_trajectory[:, :-1])
                index += 1
        self.behavior_embeddings[:] = normalize(self.behavior_embeddings)"""
        for pop in [self.offspring_pool, self.elites]:
            for individual in pop:
                self.behavior_embeddings[index, :, :, :] = individual.probabilities_for(self.sampled_trajectory[:, :-1])

        self.behavior_embeddings[:] = normalize(self.behavior_embeddings)

        """for individual_index in range(self.pop_size+self.n_offspring):
            distance = 0.
            if individual_index not in failed_indexes:
                for individual2_index in range(self.pop_size+self.n_offspring):
                    if individual_index != individual2_index and individual2_index not in failed_indexes:
                        distance += 1. - policy_similarity(self.behavior_embeddings[individual_index],
                                                           self.behavior_embeddings[individual2_index],
                                                           self.similarity_l)
            distance /= (self.pop_size + self.n_offspring - 1)
            self.perf_and_uniqueness[1, individual_index] = distance"""

    def compute_div_scores(self):
        for i in range(self.top_k+self.n_offspring):
            for j in range(self.top_k+self.n_offspring):
                if i==j:
                    self.policy_kernel_p1[i, j] = 1.
                elif j>i:
                    self.policy_kernel_p1[i, j] = policy_similarity(self.behavior_embeddings[i], self.behavior_embeddings[j],
                                                                 l=self.similarity_l)
                else:
                    self.policy_kernel_p1[i,j] = self.policy_kernel_p1[j,i]

        for index in range(self.top_k+self.n_offspring):
            div = np.linalg.det(np.delete(np.delete(self.policy_kernel_p1, index, axis=0), index, axis=1))
            if index < self.n_offspring:
                self.offspring_pool[index].div_score = 1-div
            else:
                self.elites[index-self.n_offspring].div_score = 1-div
            # self.perf_and_uniqueness[1, index, 0] = 1 - np.mean(self.policy_kernel_p1[:, index])

    def select(self):
        index = 0
        for i, pop in enumerate([self.population, self.offspring_pool, self.elites]):
            for individual in pop:
                if i==1 and individual.performance < np.min(self.perf_and_uniqueness[0,:self.pop_size,0]):
                    self.perf_and_uniqueness[:, index, 0] = -np.inf
                else:
                    self.perf_and_uniqueness[0, index, 0] = individual.performance
                    if i != 2:
                        self.perf_and_uniqueness[1, index, 0] = individual.div_score
                index += 1


        frontiers = ND_sort(self.perf_and_uniqueness[:, :-self.top_k], epsilon=2.)
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
        for s in selected:
            if s >= self.pop_size:
                print('===============', self.perf_and_uniqueness[1, s, 0],  self.perf_and_uniqueness[1, self.pop_size+self.n_offspring:, 0], '===============')
                if self.perf_and_uniqueness[1, s, 0] > np.min(self.perf_and_uniqueness[1, self.pop_size+self.n_offspring:, 0]):
                    self.logger.info('New elite !')
                    self.elites[np.argmin(self.perf_and_uniqueness[1, self.pop_size+self.n_offspring:, 0])].inerit_from(self.offspring_pool[s-self.pop_size])

        # get stats of selection...
        full_path = 'checkpoints/' + self.running_instance_id + '/ckpt_' + str(
            self.population.checkpoint_index) + '/'
        plot_perf_uniq(self.perf_and_uniqueness[:, :, 0], selected, self.population, self.elites, full_path)

        print(self.perf_and_uniqueness[:, selected, 0])

        ##########
        # if self.pop_size in selected:
        #     self.update_top_k()
        ##########

        for new_index, individual_index in enumerate(sorted(selected)):
            if individual_index < self.pop_size:
                self.population[new_index].inerit_from(self.population[individual_index])
            else:
                self.population[new_index].inerit_from(self.offspring_pool[individual_index-self.pop_size])

    def update_top_k(self):
        scores = np.empty((self.top_k,), dtype=np.float32)
        tmp_kernel = np.empty((self.top_k+1, self.top_k+1), dtype=np.float32)
        tmp_kernel[:-1, :-1] = self.policy_kernel

        for i, index_i in enumerate(self.top_k_index):
                tmp_kernel[self.top_k, i] = policy_similarity(self.behavior_embeddings[self.pop_size],
                                                                 self.behavior_embeddings[index_i],
                                                                 l=self.similarity_l)
                tmp_kernel[i, self.top_k] = tmp_kernel[self.top_k, i]

        for current_candidate in range(self.top_k):
            scores[current_candidate] = np.linalg.det(np.delete(np.delete(tmp_kernel, current_candidate,axis=0),
                                                                current_candidate, axis=1))

        best = np.max(scores)
        ### TODO ?



    def compute_novelty(self, indexes):
        for individual_index in indexes:
            distance = 0.
            for individual2_index in indexes:
                if individual_index != individual2_index:
                    distance += 1. - policy_similarity(self.behavior_embeddings[individual_index],
                                                       self.behavior_embeddings[individual2_index],
                                                       self.similarity_l)
            distance /= np.float32((len(indexes)-1))
            self.perf_and_uniqueness[1, individual_index] = distance

    def iterative_select(self):
        index = 0
        for pop in [self.population, self.offspring_pool]:
            for individual in pop:
                self.perf_and_uniqueness[0, index, 0] = individual.performance
                index += 1

        selected = list(range(0, self.pop_size))
        for index in range(self.pop_size, self.pop_size+self.n_offspring):
            selected.append(index)
            self.compute_novelty(selected)
            frontiers = ND_sort(self.perf_and_uniqueness[:, selected])
            print(frontiers)
            if len(frontiers[-1]) == 1:
                frontiers.pop()
            else:
                 frontiers[-1] = list(sorted(frontiers[-1], key=lambda index: self.perf_and_uniqueness[0, index, 0]))[1:]

            selected = [selected[index] for frontier in frontiers for index in frontier]

        self.compute_novelty(list(range(self.pop_size+self.n_offspring)))

        # get stats of selection...
        full_path = 'checkpoints/' + self.running_instance_id + '/ckpt_' + str(
            self.population.checkpoint_index) + '/'
        plot_perf_uniq(self.perf_and_uniqueness[:, :, 0], selected, self.population, full_path)

        print(self.perf_and_uniqueness[:, selected, 0])

        for new_index, individual_index in enumerate(sorted(selected)):
            if individual_index < self.pop_size:
                self.population[new_index].inerit_from(self.population[individual_index])
            else:
                self.population[new_index].inerit_from(self.offspring_pool[individual_index - self.pop_size])




    def load(self, ckpt_path):
        self.logger.info('Loading checkpoint %s ...' % ckpt_path)
        self.population.load(ckpt_path)
        self.elites.load(ckpt_path + 'elites/')
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
        self.elites.save(full_path+'elites/')
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
                        if '.pkl' in f:
                            os.remove(ckpt_path + oldest + '/' + d + '/' + f)
                    os.rmdir(ckpt_path + oldest + '/' + d)
            try:
                os.rmdir(ckpt_path + oldest)
            except Exception:
                self.logger.warning("Tried to delete a non empty checkpoint directory")


if __name__ == '__main__':
    fire.Fire(Hub)
