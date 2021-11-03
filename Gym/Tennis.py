import gym
import numpy as np


class Tennis(EnvUtil):
    def __init__(self, name):
        self.name = name
        super(Tennis, self).__init__(name)

        self['ram_locations'] = dict(enemy_x=27,
                                     enemy_y=25,
                                     enemy_score=70,
                                     ball_x=16,
                                     ball_y=15,
                                     player_x=26,
                                     player_y=24,
                                     player_score=69,
                                     ball_height=17)

        self.indexes = np.array([value for value in self['ram_locations'].values()], dtype=np.int32)
        self.reversed_indexes = np.array([26, 24, 70, 16, 15, 27, 25, 69, 17], dtype=np.int32)
        self.centers = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.scales = np.array([0.01, 0.01, 0.2, 0.01, 0.01, 0.01, 0.01, 0.2, 0.025], dtype=np.float32)
        self.state_dim = len(self.indexes) + 2
        self.full_state_dim = self.state_dim * 4
        self.y_bounds = (0.91, 1.48)
        # 2 - 74 75 - 148
        self.side = True
        self.max_frames = 15000
        self.frames_since_point = 0
        self.opposite_action_space = {
            0 : 0,
            1 : 1,
            2 : 5,
            3 : 3,
            4 : 4,
            5 : 2,
            6 : 8,
            7 : 9,
            8 : 6,
            9 : 7,
            10: 13,
            11: 11,
            12: 12,
            13: 10,
            14: 16,
            15: 17,
            16: 14,
            17: 15
        }

        self.points = np.array([71, 72], dtype=np.int32)
        self.top_side_points = np.array([0, 3, 4, 7, 8, 11])

        self.max_quality = 0

        self['objectives'] = [
            Objective('game_score'),
            Objective('aim_quality', domain=(0., 0.72)),
            Objective('mobility', domain=(0., 0.09)),
        ]

        self.action_space_dim = 9
        self.goal_dim = len(self['objectives'])

        self['problems']['SOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][0])
        self['problems']['SOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   sum_=True)

        self['problems']['MOP1']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems']['MOP2']['behavior_functions'] = self.build_objective_func(self['objectives'][1],
                                                                                   self['objectives'][2],
                                                                                   prioritized=self['objectives'][
                                                                                       0])
        self['problems']['MOP2']['complexity'] = 2

        self['problems']['MOP3']['behavior_functions'] = self.build_objective_func(self['objectives'][0],
                                                                                   self['objectives'][1],
                                                                                   self['objectives'][2])

        self['problems'].update({
            'SOP3': {
                'is_single'         : True,
                'complexity'        : 1,
                'behavior_functions': self.build_objective_func(self['objectives'][0],
                                                                self['objectives'][1],
                                                                sum_=True),
            }})

        self['problems'].update({
            'MOP4': {
                'is_single'         : True,
                'complexity'        : 1,
                'behavior_functions': self.build_objective_func(self['objectives'][0], self['objectives'][1]),
            }})

        self.mins = [np.inf, np.inf]
        self.maxs = [-np.inf, -np.inf]
        self.ball_max = [-np.inf, -np.inf]
        self.ball_min = [np.inf, np.inf]

    def action_to_id(self, action_id):
        # ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE',
        # 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
        if action_id > 1:
            action_id += 8
        else:
            action_id = 0

        return action_id

    def preprocess(self, obs):

        if self.side:
            indexes = self.indexes
        else:
            indexes = self.reversed_indexes

        reduced = (obs[indexes] - self.centers) * self.scales
        distance_from_ball = self.distance_from_ball(reduced)
        return np.concatenate(
            [(obs[indexes] - self.centers) * self.scales, [np.float32(self.side), distance_from_ball]])

    def distance_from_ball(self, obs):
        return np.sqrt((obs[3] - obs[5]) ** 2 + (obs[4] - obs[6]) ** 2) * 0.9

    def is_back(self, obs):
        # print(np.sqrt((obs[5]-obs[3])**2+(obs[6]-obs[4])**2))
        # for ob, k in zip(obs, self['ram_locations'].keys()):
        #    print(k, ob)
        # print()
        if self.side:
            return obs[6] > 1.022
        else:
            return obs[6] < 0.028

    def is_front(self, obs):
        # print(obs[3]*100, obs[4]*100)
        if self.side:
            return obs[6] < 0.756
        else:
            return obs[6] > 0.294

    def distance_ran(self, obs, obs_):
        d = np.sqrt((obs[1] - obs_[1]) ** 2 + (obs[0] - obs_[0]) ** 2)
        if d > 20 * 0.01:
            d = 0
        return d

    def self_dy(self, full_obs):
        dx = np.abs(full_obs[5] - full_obs[-self.state_dim + 5])
        dy = np.abs(full_obs[6] - full_obs[-self.state_dim + 6])
        d = dx + dy
        if d > 30 * 0.01:
            dy = 0
        return dy

    def aim_quality(self, full_obs):
        ball_x = full_obs[-self.state_dim + 3]
        ball_y = full_obs[-self.state_dim + 4]
        dplayer_y = full_obs[-self.state_dim + 1] - full_obs[-self.state_dim + 6]
        # print('1', ball_x, ball_y)
        # print('2', full_obs[-2*self.state_dim+3], full_obs[-2*self.state_dim+4])
        vector = complex(ball_y - full_obs[-2 * self.state_dim + 4], ball_x - full_obs[-2 * self.state_dim + 3])
        vector_p2p = complex(dplayer_y, full_obs[-self.state_dim + 0] - full_obs[-self.state_dim + 5])
        if self.side:
            vector *= -1
            vector_p2p *= -1
        angle = np.angle(vector)
        angle_2 = np.angle(vector_p2p)
        # print(full_obs[-self.state_dim+1]-full_obs[-self.state_dim+6])

        quality = np.clip((angle - angle_2) ** 2 * 2 + 0.1, 0, 2.25)
        if np.abs(dplayer_y) < 0.35:
            quality *= 0.2

        # opp_x = full_obs[-self.state_dim]
        # opp_y = full_obs[-self.state_dim+1]
        # dY = opp_y - ball_y

        # scale = (0.5 + 0.2 * np.abs(dY)) * np.sign(dY)
        # deviation = np.tan(angle) * scale

        # quality = np.clip(np.abs(ball_x + deviation - opp_x), 0, 1) + 0.2

        return quality

    def proximity_to_front(self, obs):
        if self.side:
            return (np.abs(0.406 - (obs[6] - 0.63)) / 0.406) ** 2
        else:
            return ((obs[6] - 0.014) / 0.406) ** 2

    def proximity_to_back(self, obs):
        if self.side:
            return (np.abs(0.406 - (1.036 - obs[6])) / 0.406) ** 2
        else:
            return (np.abs(0.406 - (obs[6] - 0.014)) / 0.406) ** 2

    def is_returning(self, preprocessed_obs, opp=False):
        if opp:
            side = not self.side
        else:
            side = self.side
        d1 = preprocessed_obs[4 + self.state_dim] - preprocessed_obs[4]
        d2 = preprocessed_obs[4 - self.state_dim * 2] - preprocessed_obs[4 - self.state_dim * 3]
        d2x = preprocessed_obs[3 - self.state_dim * 2] - preprocessed_obs[3 - self.state_dim * 3]
        if abs(d2) + abs(d2x) > 0.12 or abs(d2) + abs(d2x) < 1e-6:
            return False

        d = d1 * d2

        if not side:
            d2 = -d2

        return (d <= 0 and d2 < 0)

    def win(self, obs, last_obs, eval=False):
        dself = obs[7] - last_obs[7]
        dopp = obs[2] - last_obs[2]
        dscore = np.clip(dself, 0, 1) - np.clip(dopp, 0, 1)
        return dscore

    def swap_court(self, full_obs):
        total = np.sum(full_obs[self.points])
        if total in self.top_side_points:
            self.side = True
        else:
            self.side = False

    def eval(self, player: Individual,
             env,
             action_dim,
             frame_skip,
             max_frames,
             min_games,
             render=False,
             slow_factor=0.04,
             ):

        r = {
            'game_reward'   : 0.0,
            'avg_length'    : 0.,
            'total_punition': 0.0,
            'game_score'    : 0.0,
            'entropy'       : 0.0,
            'eval_length'   : 0,
            'mobility'      : 0.,
            'n_shoots'      : 0.,
            'aim_quality'   : 0.,
            'opp_shoots'    : 0,
        }
        frame_count = 0.
        n_games = 0
        actions = [0] * env.action_space.n
        dist = np.zeros((action_dim,), dtype=np.float32)

        while n_games < min_games:
            done = False
            observation = env.reset()
            self.frames_since_point = 0

            self.swap_court(observation)
            observation = self.preprocess(observation)
            observation = np.concatenate([observation, observation, observation, observation])
            while not done and frame_count < max_frames:
                action, dist_ = player.pi.policy.get_action(observation, return_dist=True, eval=True)
                dist += dist_
                actions[action] += 1
                reward = 0
                if render:
                    env.render()
                    time.sleep(slow_factor)

                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(
                        self.action_to_id(action))
                    reward += rr
                    if done:
                        break

                while not done and np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))) < 1e-5:
                    for _ in range(frame_skip):
                        observation_, rr, done, info = env.step(
                            self.action_to_id(action))
                        reward += rr
                        if done:
                            break

                if reward == 0:
                    if abs(observation[3] - observation[3 + 3 * self.state_dim]) < 1e-4 and \
                            abs(observation[4] - observation[4 + 3 * self.state_dim]) < 1e-4:
                        self.frames_since_point += 1
                        if self.frames_since_point > 300 // frame_skip:
                            r['game_score'] = -np.inf
                            r['mobility'] = -np.inf
                            r['aim_quality'] = -np.inf
                            return r
                else:
                    self.frames_since_point = 0

                self.swap_court(observation_)
                observation_ = self.preprocess(observation_)
                r['game_score'] += reward  # self.win(observation_, observation[len(observation) * 3 // 4:]) * 100
                # r['opponent_run_distance'] += self.distance_ran(observation[3 * len(observation) // 4:], observation_)
                observation = np.concatenate([observation[len(observation) // 4:], observation_])
                r['mobility'] += self.self_dy(observation)

                is_returning = self.is_returning(observation)
                if is_returning:
                    r['n_shoots'] += 1
                    r['aim_quality'] += self.aim_quality(observation)
                r['opp_shoots'] += int(self.is_returning(observation, True))
                r['game_reward'] += reward
                if reward < 0:
                    r['total_punition'] += reward
                    r['aim_quality'] += reward * 0.05

                frame_count += 1

            n_games += 1

        print(actions)
        r['avg_length'] = frame_count / float(n_games)
        r['game_score'] = (r['game_score'] + 24. * n_games) / (48. * n_games)
        dist /= float(frame_count)
        r['entropy'] = -np.sum(np.log(dist + 1e-8) * dist)
        r['eval_length'] = frame_count
        r['aim_quality'] = np.clip(r['aim_quality'], 0, np.inf)
        r['aim_quality'] /= np.clip(r['n_shoots'], 48 * n_games, np.inf)
        r['mobility'] /= frame_count

        return r

    def play(self, player: Individual,
             env,
             batch_index,
             traj_length,
             frame_skip,
             trajectory,
             action_dim,
             observation=None,
             gpu=-1):

        actions = [0] * action_dim
        force_reset = False

        if observation is None:
            observation = env.reset()
            self.frames_since_point = 0
            self.swap_court(observation)
            observation = self.preprocess(observation)
            observation = np.concatenate([observation, observation, observation, observation])

        for frame_count in range(traj_length):
            action = player.pi.policy.get_action(observation, gpu=gpu)
            actions[action] += 1
            reward = 0

            # env.render()
            # time.sleep(0.5)
            for _ in range(frame_skip):
                observation_, rr, done, info = env.step(
                    self.action_to_id(action))
                reward += rr
                if done:
                    break
            while not done and np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))) < 1e-5:
                for _ in range(frame_skip):
                    observation_, rr, done, info = env.step(
                        self.action_to_id(action))
                    reward += rr
                    if done:
                        break

                # print(np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))) < 1e-5,
                #      np.max(np.abs(observation[-self.state_dim:] - self.preprocess(observation_))))

            # print(observation_)
            self.swap_court(observation_)
            punish = 0
            if reward == 0:
                if abs(observation[3] - observation[3 + 3 * self.state_dim]) < 1e-4 and \
                        abs(observation[4] - observation[4 + 3 * self.state_dim]) < 1e-4:
                    self.frames_since_point += 1
                    if self.frames_since_point > 200 // frame_skip:
                        punish -= 5
                        force_reset = True
            else:
                self.frames_since_point = 0

            observation_ = self.preprocess(observation_)
            # win = self.win(observation_, observation[len(observation) * 3 // 4:]) * 100
            # front = np.clip(self.proximity_to_front(observation_) - 0.25, 0, 1)
            back = self.proximity_to_back(observation_)

            trajectory['state'][batch_index, frame_count] = observation
            trajectory['action'][batch_index, frame_count] = action
            trajectory['rew'][batch_index, frame_count] = reward * player.reward_weight[0] + punish
            # -0.5 * front * player.reward_weight[2]

            trajectory['base_rew'][batch_index, frame_count] = reward

            if done or force_reset:
                force_reset = False
                self.frames_since_point = 0
                observation = self.preprocess(env.reset())
                observation = np.concatenate([observation, observation, observation, observation])
            else:
                observation = np.concatenate([observation[len(observation) // 4:], observation_])
                if self.is_returning(observation):
                    aim_quality = self.aim_quality(observation)
                else:
                    aim_quality = 0
                trajectory['rew'][batch_index, frame_count] += \
                    aim_quality * player.reward_weight[1] \
                    + (1 + 2 * back) * self.self_dy(observation) * player.reward_weight[2] * 0.03

        return observation



class Tennis:

    def __init__(self, frame_skip, render=False):

        self.ram_indexes = dict(enemy_x=27,
                                     enemy_y=25,
                                     enemy_score=70,
                                     ball_x=16,
                                     ball_y=15,
                                     player_x=26,
                                     player_y=24,
                                     player_score=69,
                                     ball_height=17)

        self.indexes = np.array([value for value in self.ram_indexes.values()], dtype=np.int32)
        self.reversed_indexes = np.array([26, 24, 70, 16, 15, 27, 25, 69, 17], dtype=np.int32)
        self.centers = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.scales = np.array([0.01, 0.01, 0.2, 0.01, 0.01, 0.01, 0.01, 0.2, 0.025], dtype=np.float32)
        self.state_dim = len(self.indexes) + 2
        self.full_state_dim = self.state_dim * 4
        self.y_bounds = (0.91, 1.48)
        # 2 - 74 75 - 148
        self.side = True

        self.max_frames = 15000

        self.frames_since_point = 0

        self.points = np.array([71, 72], dtype=np.int32)
        self.top_side_points = np.array([0, 3, 4, 7, 8, 11])

        self.action_space_dim = 18

        self.env = gym.make('Tennis-ramNoFrameskip-v4')
        self.frame_skip = frame_skip
        self.render = render
        init_obs = self.preprocess(self.env.reset())
        self.obs = np.concatenate([init_obs for _ in range(4)])

    def preprocess(self, obs):

        if self.side:
            indexes = self.indexes
        else:
            indexes = self.reversed_indexes

        reduced = (obs[indexes] - self.centers) * self.scales
        distance_from_ball = self.distance_from_ball(reduced)
        return np.concatenate(
            [(obs[indexes] - self.centers) * self.scales, [np.float32(self.side), distance_from_ball]])

    def distance_from_ball(self, obs):
        return np.sqrt((obs[3] - obs[5]) ** 2 + (obs[4] - obs[6]) ** 2) * 0.9

    def distance_ran(self, obs, obs_):
        d = np.sqrt((obs[1] - obs_[1]) ** 2 + (obs[0] - obs_[0]) ** 2)
        if d > 20 * 0.01:
            d = 0
        return d

    def self_dy(self, full_obs):
        dx = np.abs(full_obs[5] - full_obs[-self.state_dim + 5])
        dy = np.abs(full_obs[6] - full_obs[-self.state_dim + 6])
        d = dx + dy
        if d > 30 * 0.01:
            dy = 0
        return dy

    def aim_quality(self, full_obs):
        ball_x = full_obs[-self.state_dim + 3]
        ball_y = full_obs[-self.state_dim + 4]
        dplayer_y = full_obs[-self.state_dim + 1] - full_obs[-self.state_dim + 6]
        # print('1', ball_x, ball_y)
        # print('2', full_obs[-2*self.state_dim+3], full_obs[-2*self.state_dim+4])
        vector = complex(ball_y - full_obs[-2 * self.state_dim + 4], ball_x - full_obs[-2 * self.state_dim + 3])
        vector_p2p = complex(dplayer_y, full_obs[-self.state_dim + 0] - full_obs[-self.state_dim + 5])
        if self.side:
            vector *= -1
            vector_p2p *= -1
        angle = np.angle(vector)
        angle_2 = np.angle(vector_p2p)
        # print(full_obs[-self.state_dim+1]-full_obs[-self.state_dim+6])

        quality = np.clip((angle - angle_2) ** 2 * 2 + 0.1, 0, 2.25)
        if np.abs(dplayer_y) < 0.35:
            quality *= 0.2

        # opp_x = full_obs[-self.state_dim]
        # opp_y = full_obs[-self.state_dim+1]
        # dY = opp_y - ball_y

        # scale = (0.5 + 0.2 * np.abs(dY)) * np.sign(dY)
        # deviation = np.tan(angle) * scale

        # quality = np.clip(np.abs(ball_x + deviation - opp_x), 0, 1) + 0.2

        return quality

    def proximity_to_front(self, obs):
        if self.side:
            return (np.abs(0.406 - (obs[6] - 0.63)) / 0.406) ** 2
        else:
            return ((obs[6] - 0.014) / 0.406) ** 2

    def proximity_to_back(self, obs):
        if self.side:
            return (np.abs(0.406 - (1.036 - obs[6])) / 0.406) ** 2
        else:
            return (np.abs(0.406 - (obs[6] - 0.014)) / 0.406) ** 2

    def is_returning(self, preprocessed_obs, opp=False):
        if opp:
            side = not self.side
        else:
            side = self.side
        d1 = preprocessed_obs[4 + self.state_dim] - preprocessed_obs[4]
        d2 = preprocessed_obs[4 - self.state_dim * 2] - preprocessed_obs[4 - self.state_dim * 3]
        d2x = preprocessed_obs[3 - self.state_dim * 2] - preprocessed_obs[3 - self.state_dim * 3]
        if abs(d2) + abs(d2x) > 0.12 or abs(d2) + abs(d2x) < 1e-6:
            return False

        d = d1 * d2

        if not side:
            d2 = -d2

        return (d <= 0 and d2 < 0)


    def swap_court(self, full_obs):
        total = np.sum(full_obs[self.points])
        if total in self.top_side_points:
            self.side = True
        else:
            self.side = False


    def full_reset(self):
        self.env.close()
        self.__init__(self.frame_skip, self.render)

    def reset(self):
        self.side = True
        self.frames_since_point = 0
        obs = self.preprocess(self.env.reset())
        self.obs = np.concatenate([obs for _ in range(4)])

    def step(self, action):

        reward = 0
        done = False
        full_obs = None
        if self.render:
            self.env.render()
        for _ in range(self.frame_skip):
            full_obs, rr, done, info = self.env.step(action)
            reward += rr
            if done:
                break
        obs = self.preprocess(full_obs)
        while not done and np.max(np.abs(self.obs[-self.state_dim:] - obs)) < 1e-5:
            full_obs, rr, done, info = self.env.step(action)
            obs = self.preprocess(full_obs)
            reward += rr
            if done:
                break

        if done :
            self.env.reset()

        self.swap_court(full_obs)
        if reward == 0:
            if abs(self.obs[3] - self.obs[3 + 3 * self.state_dim]) < 1e-4 and \
                    abs(self.obs[4] - self.obs[4 + 3 * self.state_dim]) < 1e-4:
                self.frames_since_point += 1
                if self.frames_since_point > 200 // self.frame_skip:
                    reward -= 5
                    self.env.reset()

        else:
            self.frames_since_point = 0

        self.obs[:self.state_dim] = self.obs[self.state_dim:]
        self.obs[-self.state_dim:] = obs

        return done, reward




