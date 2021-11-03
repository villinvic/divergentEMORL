import gym
import numpy as np





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




