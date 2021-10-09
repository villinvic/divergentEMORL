import zmq
import pandas as pd
import numpy as np
import tensorflow as tf
from time import time
import datetime
import fire

from EMORL.Individual import Individual
from Game.core import Game
from Game.rewards import Rewards
from config.Loader import Default


class Hub(Default):
    def __init__(self, ip='127.0.0.1'):
        super(Hub, self).__init__()
        dummy_env = Game()
        self.agent = Individual(0, dummy_env.state_dim, dummy_env.action_dim, [], trainable=True)
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
        log_dir = 'logs/train' + datetime.datetime.now().strftime("EMORL_%Y-%m-%d_%H-%M")
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

    def pub_params(self):
        try:
            # self.param_socket.send_string(str(individual.id), zmq.SNDMORE)
            # self.param_socket.send_pyobj(individual.get_arena_genes(), flags=zmq.NOBLOCK)
            self.blob_socket.send_pyobj(self.agent.get_arena_genes())
        except zmq.ZMQError:
            pass

    def train(self):
        if len(self.exp) >= self.BATCH_SIZE:
            # Get experience from the queue
            trajectory = pd.DataFrame(self.exp[:self.BATCH_SIZE]).values
            self.exp = self.exp[self.BATCH_SIZE:]

            # Cook data
            states = np.float32(np.stack(trajectory[:, 0], axis=0))
            actions = np.float32(np.stack(trajectory[:, 1], axis=0)[:, :-1])
            probs = np.float32(np.stack(trajectory[:, 2], axis=0)[:, :-1])
            wins = np.float32(np.stack(trajectory[:, 3], axis=0)[:, :-1])
            rews = self.rewards.compute(states, self.agent.genotype['experience'], wins)
            # Train
            with tf.summary.record_if(self.train_cntr % self.write_summary_freq == 0):

                self.agent.mean_entropy = \
                    self.agent.genotype['brain'].train(self.agent.genotype['learning'],states, actions, rews, probs, 0)
            self.train_cntr += 1
            tf.summary.experimental.set_step(self.train_cntr)

            print('train !', self.agent.mean_entropy)

            return True

        return False

    def __call__(self):
        try:
            self.pub_params()
            last_pub_time = time()

            while True:
                self.recv_training_data()
                self.train()

                current = time()
                if current - last_pub_time > 10:
                    print('pub')
                    self.pub_params()
                    last_pub_time = current

        except KeyboardInterrupt:
            print('Hub exited.')

    # Init pop
    # select p pairs of individuals, operate crossover with low prob, mutation
    # for each individual, select k random (or nearest ?)  individual behavior stats as a landmark for divergence
    # Train each individuals for x minutes, 1 by 1, on the trainer (can be the Hub, with GPU)
    # use recent training eval for selection
    # select regarding performance and behavior uniqueness

if __name__ == '__main__':
    fire.Fire(Hub)