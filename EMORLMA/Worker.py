import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import fire
import zmq
import time
import signal

from config.Loader import Default
from Gym.BoxingMA import BoxingMA as Game
from EMORLMA.Individual import Individual


class Worker(Default):
    def __init__(self, render=False, hub_ip='127.0.0.1'):
        self.render = render
        super(Worker, self).__init__()
        self.game =  Game(render=render, frameskip=self.frameskip)
        self.players = (Individual(-1, self.game.state_dim, self.game.action_dim, []),
                        Individual(-1, self.game.state_dim, self.game.action_dim, []))

        c = zmq.Context()
        self.blob_socket = c.socket(zmq.SUB)
        self.blob_socket.subscribe(b'')
        self.blob_socket.connect("tcp://%s:%d" % (hub_ip, self.PARAM_PORT))
        self.exp_socket = c.socket(zmq.PUSH)
        self.exp_socket.connect("tcp://%s:%d" % (hub_ip, self.EXP_PORT))

        hidden_h, hidden_c = self.player.genotype['brain'].init_body(np.zeros((1,1, self.game.state_dim), dtype=np.float32))


        self.trajectories = tuple({
            'state' : np.zeros((self.TRAJECTORY_LENGTH, self.game.state_dim), dtype=np.float32),
            'action' : np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.int32),
            'probs': np.zeros((self.TRAJECTORY_LENGTH, self.game.action_dim), dtype=np.float32),
            'win': np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.float32),
            'hidden_states': np.zeros((2, 128), dtype=np.float32),
            'id': None,
        } for _ in range(2))

        self.trajectories[0]['hidden_states'][:] = np.concatenate([hidden_h, hidden_c], axis=0)
        self.trajectories[1]['hidden_states'][:] = np.concatenate([hidden_h, hidden_c], axis=0)

        signal.signal(signal.SIGINT, lambda frame, signal : sys.exit())

    def get_params(self):
        try:
            for p in self.players:
                p.set_arena_genes(self.blob_socket.recv_pyobj(zmq.NOBLOCK))
                # get genes depending on player id
        except zmq.ZMQError:
            return False
            pass
        return True

    def send_exp(self):
        self.exp_socket.send_pyobj(self.trajectories[0])
        self.exp_socket.send_pyobj(self.trajectories[1])

    def play_trajectory(self):
        for index in range(self.TRAJECTORY_LENGTH):
            if self.render:
                self.game.render()
            #s = self.game.state / self.game.scales
            action_id, distribution, hidden_h, hidden_c = self.players[0].policy(self.game.state)
            action_id_opp, distribution_opp, hidden_h_opp, hidden_c_opp = self.players[1].policy(self.game.opp_state)
            self.trajectories[0]['state'][index, :] = self.game.state
            self.trajectories[0]['action'][index] = action_id
            self.trajectories[0]['probs'][index] = distribution

            self.trajectories[1]['state'][index, :] = self.game.opp_state
            self.trajectories[2]['action'][index] = action_id_opp
            self.trajectories[3]['probs'][index] = distribution_opp

            done, win = self.game.step([action_id, action_id_opp])
            #perf, done, reward = self.game.step(action_id)
            self.trajectories[0]['win'][index] = win
            self.trajectories[1]['win'][index] = -win
            #self.trajectory['reward'][index] = reward

            if done :
                self.game.reset()
                for p in self.players:
                    p.genotype['brain'].lstm.reset_states()
        self.send_exp()

        self.trajectories[0]['hidden_states'][:] = np.concatenate([hidden_h,hidden_c], axis=0)
        self.trajectories[1]['hidden_states'][:] = np.concatenate([hidden_h_opp, hidden_c_opp], axis=0)

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
            c += 1
            self.get_params()


if __name__ == '__main__':
    fire.Fire(Worker)