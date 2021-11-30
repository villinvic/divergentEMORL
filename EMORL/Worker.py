import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import fire
import zmq
import time
import signal

from config.Loader import Default
#from Game.core import Game
from Gym.Boxing import Boxing as Game
from EMORL.Individual import Individual



class Worker(Default):
    def __init__(self, render=False, hub_ip='127.0.0.1'):
        self.render = render
        super(Worker, self).__init__()
        self.game =  Game(render=render, frameskip=self.frameskip)
        self.player = Individual(-1, self.game.state_dim, self.game.action_dim, [])
        #self.player = Individual(-1, self.game.env.observation_space.shape[0], self.game.env.action_space.shape[0], [])

        c = zmq.Context()
        self.blob_socket = c.socket(zmq.SUB)
        self.blob_socket.subscribe(b'')
        self.blob_socket.connect("tcp://%s:%d" % (hub_ip, self.PARAM_PORT))
        self.exp_socket = c.socket(zmq.PUSH)
        self.exp_socket.connect("tcp://%s:%d" % (hub_ip, self.EXP_PORT))

        hidden_h, hidden_c = self.player.genotype['brain'].init_body(np.zeros((1,1, self.game.state_dim), dtype=np.float32))


        self.trajectory = {
            'state' : np.zeros((self.TRAJECTORY_LENGTH, self.game.state_dim), dtype=np.float32),
            'action' : np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.int32),
            'probs': np.zeros((self.TRAJECTORY_LENGTH, self.game.action_dim), dtype=np.float32),
            'win': np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.float32),
            #'reward': np.zeros((self.TRAJECTORY_LENGTH, self.game.n_rewards), dtype=np.float32),
            'hidden_states': np.zeros((2, 128), dtype=np.float32),
        }

        self.trajectory['hidden_states'][:] = np.concatenate([hidden_h,hidden_c], axis=0)

        signal.signal(signal.SIGINT, lambda frame, signal : sys.exit())

    def get_params(self, block=False):
        try:
            if block :
                self.player.set_arena_genes(self.blob_socket.recv_pyobj())
            else:
                self.player.set_arena_genes(self.blob_socket.recv_pyobj(zmq.NOBLOCK))
        except zmq.ZMQError:
            return False
            pass
        return True

    def send_exp(self):
        self.exp_socket.send_pyobj(self.trajectory)

    def play_trajectory(self):
        for index in range(self.TRAJECTORY_LENGTH):
            if self.render:
                self.game.render()
            #s = self.game.state / self.game.scales
            action_id, distribution, hidden_h, hidden_c = self.player.policy(self.game.state)
            self.trajectory['state'][index, :] = self.game.state
            self.trajectory['action'][index] = action_id
            self.trajectory['probs'][index] = distribution
            done, win = self.game.step(action_id)
            #perf, done, reward = self.game.step(action_id)
            self.trajectory['win'][index] = win
            #self.trajectory['reward'][index] = reward

            if done :
                self.game.reset()
                if self.player.genotype['brain'].has_lstm:
                    self.player.genotype['brain'].lstm.reset_states()
        self.send_exp()

        self.trajectory['hidden_states'][:] = np.concatenate([hidden_h,hidden_c], axis=0)

    def __call__(self):
        for _ in range(3):
            x = self.get_params()
            if x :
                break
            time.sleep(1)

        print('LETS GO')
        c = 0
        while True:
            self.play_trajectory()
            c += 1
            self.get_params()


if __name__ == '__main__':
    fire.Fire(Worker)