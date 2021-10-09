import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import fire
import zmq
import time
import signal

from config.Loader import Default
from Game.core import Game
from EMORL.Individual import Individual


class Worker(Default):
    def __init__(self, render=False, hub_ip='127.0.0.1'):
        self.render = render
        super(Worker, self).__init__()
        self.game = Game(render)
        self.player = Individual(-1, self.game.state_dim, self.game.action_dim, [])


        c = zmq.Context()
        self.blob_socket = c.socket(zmq.SUB)
        self.blob_socket.subscribe(b'')
        self.blob_socket.connect("tcp://%s:%d" % (hub_ip, self.PARAM_PORT))
        self.exp_socket = c.socket(zmq.PUSH)
        self.exp_socket.connect("tcp://%s:%d" % (hub_ip, self.EXP_PORT))


        self.trajectory = {
            'state' : np.zeros((self.TRAJECTORY_LENGTH, self.game.state_dim), dtype=np.float32),
            'action' : np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.int32),
            'probs': np.zeros((self.TRAJECTORY_LENGTH, self.game.action_dim), dtype=np.float32),
            'win': np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.float32),
        }

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
        win = 0
        for index in range(self.TRAJECTORY_LENGTH):
            if self.render:
                self.game.render()
            s = self.game.state / self.game.scales
            action_id, distribution, hidden_h, hidden_c = self.player.policy(s)
            self.trajectory['state'][index, :] = s
            self.trajectory['action'][index] = action_id
            self.trajectory['probs'][index] = distribution
            self.trajectory['win'][index] = win
            done, win = self.game.step(action_id)

            if done :
                self.game.reset()

    def __call__(self):
        for _ in range(30):
            x = self.get_params()
            if x :
                break
            time.sleep(1)

        print('LETS GO')
        c = 0
        while True:
            self.play_trajectory()
            self.send_exp()
            c += 1
            self.get_params()


if __name__ == '__main__':
    fire.Fire(Worker)