import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import fire
import zmq
import time
import threading
import signal
from collections import deque

from config.Loader import Default
#from Game.core import Game
from Gym.Boxing import Boxing as Game
#from Gym.Kfm import Kfm as Game
from Melee.game.console import Console
from Melee.characters.characters import string2char
from Melee.input.pad import Button
from Melee.game.state import GameState
from Melee.input.pad import Pad
from EMORL.Individual import Individual



class Worker(Default):
    def __init__(self, render=False, hub_ip='127.0.0.1'):
        self.render = render
        super(Worker, self).__init__()
        self.game = Game(render=render, frameskip=self.frameskip)
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

        if self.player.genotype['brain'].has_lstm:
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

        if self.player.genotype['brain'].has_lstm:
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



class MeleeWorker(Default):
    def __init__(self, ID, render=False, hub_ip='127.0.0.1'):
        self.render = render
        self.ID = ID
        super(MeleeWorker, self).__init__()
        self.game = Console(ID, render)
        self.player = Individual(-1, self.game.state_dim, self.game.action_dim, [])

        self.pad = Pad(self.pad_path + str(ID) + '/', player_id=0)
        self.last_action_id = 0
        self.next_possible_move_frame = 0
        self.is_dead = False
        self.trajectory_index = 0
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

        if self.player.genotype['brain'].has_lstm:
            self.trajectory['hidden_states'][:] = np.concatenate([hidden_h,hidden_c], axis=0)

        signal.signal(signal.SIGINT, self.exit)

        self.chars = (string2char[self.char], string2char[self.opp_char])
        self.game.state.update_players(self.chars)
        self.action_space = self.chars[0].action_space
        self.action_queue = deque()

    def press_A(self):
        self.pad.press_button(Button.A)

    def update_death(self):
        self.is_dead = self.trajectory['state'][self.trajectory_index % self.TRAJECTORY_LENGTH][
                           GameState.stock_indexes[0]] == 0

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

    def act(self, state):
        if not self.action_queue and not self.is_dead:
            traj_index = self.trajectory_index % self.TRAJECTORY_LENGTH
            state.get(self.trajectory['state'][traj_index], 0, self.last_action_id)
            self.update_death()

            action_id, distribution, hidden_h, hidden_c = self.player.policy(self.trajectory['state'][traj_index])
            if action_id >= self.game.action_dim:
                print(distribution)
                action_id = self.game.action_dim - 1
            action = self.action_space[action_id]
            if isinstance(action, list):
                self.action_queue.extend(reversed(action))
            else:
                self.action_queue.append(action)
            self.last_action_id = action_id
            self.trajectory['action'][traj_index] = action_id
            self.trajectory['probs'][traj_index] = distribution

            if traj_index == 0:
                #self.trajectory['hidden_states'][:] = hidden_h, hidden_c

                if self.trajectory_index > 0:
                    self.send_exp()
                    self.get_params()

            self.trajectory_index += 1

        if not self.is_dead and state.frame >= self.next_possible_move_frame:
            action = self.action_queue.pop()
            self.next_possible_move_frame = state.frame + action['duration']
            action.send_controller(self.pad)

    def finalize(self, state):
        traj_index = self.trajectory_index % self.TRAJECTORY_LENGTH
        if traj_index > 0:
            state.get(self.trajectory['state'][traj_index], 0, self.last_action_id)
            if traj_index < self.TRAJECTORY_LENGTH - 1:
                self.trajectory['state'][traj_index + 1:] = self.trajectory['state'][traj_index]
            self.trajectory_index = 0

            self.send_exp()
            self.get_params()

        self.action_queue.clear()
        self.next_possible_move_frame = -np.inf
        self.is_dead = False

    def __call__(self):
        for _ in range(3):
            x = self.get_params()
            if x :
                break
            time.sleep(1)

        print('LETS GO')
        c = 0
        while True:
            self.play_game()


    def play_game(self):

        self.game.state.mw.bind()
        t = threading.Thread(target=self.pad.connect)
        t.start()
        self.game.dolphin.run(*self.chars)

        frames = 0
        done = False
        result = None
        self.game.state.init()
        while not done and frames < 7*60*60:
            # Player hold a trajectory instance
            # choose action based on state, send to controller
            # data required : time of traj, states, actions, probabilities
            done, result = self.game.state.update()
            if result is None:
                self.finalize(self.game.state)
                self.game.dolphin.close()
                self.game.state.mw.unbind()
                self.pad.unbind()
                self.pad.pipe.close()
                time.sleep(1)
                return self.play_game()

            self.act(self.game.state)
            frames += 1

        self.finalize(self.game.state)
        self.game.dolphin.close()
        self.game.state.mw.unbind()
        self.pad.unbind()
        self.pad.pipe.close()

        return result

    def exit(self, frame, signal):
        self.game.close()
        self.pad.unbind()
        self.pad.pipe.close()
        print('Arena %d closed' % self.ID)
        sys.exit(0)


if __name__ == '__main__':
    fire.Fire(Worker)
