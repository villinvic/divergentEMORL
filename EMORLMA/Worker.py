import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import fire
import zmq
import zmq.decorators
import time
import signal

from config.Loader import Default
#from Game.core import Game
from Gym.BoxingMA import BoxingMA as Game
from EMORLMA.Individual import Individual



class Worker(Default):
    def __init__(self, render=False, hub_ip='127.0.0.1'):
        self.render = render
        self.hub_ip = hub_ip
        super(Worker, self).__init__()
        self.game = Game(render=render, frameskip=self.frameskip)
        self.players = (Individual(-1, self.game.state_dim, self.game.action_dim, []),
                        Individual(-1, self.game.state_dim, self.game.action_dim, []))
        #self.player = Individual(-1, self.game.env.observation_space.shape[0], self.game.env.action_space.shape[0], [])
        c = zmq.Context()
        self.exp_socket = c.socket(zmq.PUSH)
        self.exp_socket.connect("tcp://%s:%d" % (hub_ip, self.EXP_PORT))
        self.player_ids = np.zeros(2, dtype=np.int32)

        for p in self.players:
            hidden_h, hidden_c = p.genotype['brain'].init_body(np.zeros((1,1, self.game.state_dim), dtype=np.float32))


        self.trajectory = [{
            'state' : np.zeros((self.TRAJECTORY_LENGTH, self.game.state_dim), dtype=np.float32),
            'action' : np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.int32),
            'probs': np.zeros((self.TRAJECTORY_LENGTH, self.game.action_dim), dtype=np.float32),
            'win': np.zeros((self.TRAJECTORY_LENGTH,), dtype=np.float32),
            #'reward': np.zeros((self.TRAJECTORY_LENGTH, self.game.n_rewards), dtype=np.float32),
            'hidden_states': np.zeros((2, 128), dtype=np.float32),
            'player_id': 0
        } for _ in range(2)]

        for i, p in enumerate(self.players):
            if p.genotype['brain'].has_lstm:
                self.trajectory[i]['hidden_states'][:] = np.concatenate([hidden_h,hidden_c], axis=0)

        signal.signal(signal.SIGINT, lambda frame, signal : sys.exit())

    @zmq.decorators.socket(zmq.REQ)
    def request_match(self, socket, last_match_result=None):
        socket.connect("tcp://%s:%d" % (self.hub_ip, self.PARAM_PORT))
        socket.setsockopt(zmq.RCVTIMEO, 50000)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.send_pyobj((last_match_result, self.player_ids))
            params, player_ids = socket.recv_pyobj()
            for param, player in zip(params, self.players):
                player.set_arena_genes(param)
            for i, player_id in enumerate(player_ids):
                self.trajectory[i]['player_id'] = player_id
            self.player_ids[:] = player_ids
        except zmq.ZMQError as e:
            print(e)
            return False

        return True

    def send_exp(self):
        if self.player_ids[0] == 20:
            self.exp_socket.send_pyobj(self.trajectory[0])
        if self.player_ids[1] == 20:
            self.exp_socket.send_pyobj(self.trajectory[1])

        # TODO LSTM hidden state update

    def play_match(self):
        done = False
        index = 0
        while not done:
            action_1, distribution_1, hidden_h_1, hidden_c_1 = self.players[0].policy(self.game.state)
            action_2, distribution_2, hidden_h_2, hidden_c_2 = self.players[1].policy(self.game.opp_state)
            self.trajectory[0]['state'][index % self.TRAJECTORY_LENGTH, :] = self.game.state
            self.trajectory[0]['action'][index % self.TRAJECTORY_LENGTH] = action_1
            self.trajectory[0]['probs'][index % self.TRAJECTORY_LENGTH] = distribution_1

            self.trajectory[1]['state'][index % self.TRAJECTORY_LENGTH, :] = self.game.opp_state
            self.trajectory[1]['action'][index % self.TRAJECTORY_LENGTH] = action_2
            self.trajectory[1]['probs'][index % self.TRAJECTORY_LENGTH] = distribution_2

            done, win = self.game.step([action_1, action_2])
            self.trajectory[0]['win'][index % self.TRAJECTORY_LENGTH] = win
            self.trajectory[1]['win'][index % self.TRAJECTORY_LENGTH] = -win

            index += 1

            if index % self.TRAJECTORY_LENGTH == 0:
                self.send_exp()
        if index % self.TRAJECTORY_LENGTH != 0:
            for player_index in range(2):
                self.trajectory[player_index]['state'][index % self.TRAJECTORY_LENGTH:, :] =\
                    self.trajectory[player_index]['state'][index % self.TRAJECTORY_LENGTH-1, :]
                self.trajectory[player_index]['win'][index % self.TRAJECTORY_LENGTH:] = 0.
            self.send_exp()

        self.game.reset()
        for p in self.players:
            if p.genotype['brain'].has_lstm:
                p.genotype['brain'].lstm.reset_states()

        return win


    def __call__(self):
        for _ in range(10):
            x = self.request_match(last_match_result=None)
            if x :
                break
            time.sleep(1)

        print('LETS GO')
        c = 0
        while True:
            last_match_result = self.play_match()
            c += 1
            print(last_match_result, c)
            while not self.request_match(last_match_result=last_match_result):
                print(c, 'stuck ?')
                pass


if __name__ == '__main__':
    fire.Fire(Worker)