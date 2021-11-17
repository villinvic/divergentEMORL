import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.backend import set_value
import numpy as np
from tensorflow.keras.activations import relu, softmax
from copy import deepcopy

from config.Loader import Default
from EMORL.misc import nn_crossover

class Distribution(object):
    def __init__(self, dim):
        self._dim = dim
        self._tiny = 1e-8

    @property
    def dim(self):
        raise self._dim

    def kl(self, old_dist, new_dist):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio(self, x, old_dist, new_dist):
        raise NotImplementedError

    def entropy(self, dist):
        raise NotImplementedError

    def log_likelihood_sym(self, x, dist):
        raise NotImplementedError

    def log_likelihood(self, xs, dist):
        raise NotImplementedError


class Categorical(Distribution):
    def kl(self, old_prob, new_prob):
        """
        Compute the KL divergence of two Categorical distribution as:
            p_1 * (\log p_1  - \log p_2)
        """
        return tf.reduce_sum(
            old_prob * (tf.math.log(old_prob + self._tiny) - tf.math.log(new_prob + self._tiny)))

    def likelihood_ratio(self, x, old_prob, new_prob):
        return (tf.reduce_sum(new_prob * x) + self._tiny) / (tf.reduce_sum(old_prob * x) + self._tiny)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            \log \sum(p_i * x_i)

        :param x (tf.Tensor or np.ndarray): Values to compute log likelihood
        :param param (Dict): Dictionary that contains probabilities of outputs
        :return (tf.Tensor): Log probabilities
        """
        probs = param["prob"]
        assert probs.shape == x.shape, \
            "Different shape inputted. You might have forgotten to convert `x` to one-hot vector."
        return tf.math.log(tf.reduce_sum(probs * x, axis=1) + self._tiny)

    def sample(self, probs, amount=1):
        # NOTE: input to `tf.random.categorical` is log probabilities
        # For more details, see https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/random/categorical
        # [probs.shape[0], 1]
        # tf.print(probs, tf.math.log(probs), tf.random.categorical(tf.math.log(probs), amount), summarize=-1)
        return tf.cast(tf.map_fn(lambda p: tf.cast(tf.random.categorical(tf.math.log(p), amount), tf.float32), probs),
                       tf.int64)

    def entropy(self, probs):
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)


class CategoricalActor(tf.keras.Model):
    '''
    Actor model class
    '''

    def __init__(self, action_dim, epsilon, layer_dims, default_activation='elu',
                 name="CategoricalActor"):
        super().__init__(name=name)
        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim
        self.epsilon = epsilon

        self.denses = [Dense(dim, activation=default_activation, dtype="float32", name='dense_%d' % i)
                       for i, dim in enumerate(layer_dims)]

        self.prob = Dense(action_dim, dtype='float32', name="prob", activation="softmax")

    def _compute_feature(self, features):
        for layer in self.denses:
            features = layer(features)

        return features

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """

        features = self._compute_feature(states)

        probs = self.prob(features) * (1.0 - self.epsilon) + self.epsilon / np.float32(self.action_dim)

        return probs

    def get_action(self, state):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == 1

        state = state[np.newaxis][np.newaxis].astype(
            np.float32) if is_single_state else state
        action, probs = self._get_action_body(tf.constant(state))

        return (action.numpy()[0][0], probs.numpy()) if is_single_state else (action, probs)

    @tf.function
    def _get_action_body(self, state):
        probs = self._compute_dist(state)
        action = tf.squeeze(self.dist.sample(probs), axis=1)
        return action, probs

    def get_probs(self, states):
        return self._compute_dist(states)

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)


class Policy(tf.keras.Model, Default):
    '''
    Actor model class
    '''

    def __init__(self, action_dim, layer_dims, lstm_dim, default_activation='elu',
                 name="CategoricalActor"):
        super().__init__(name=name)
        Default.__init__(self)

        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim
        self.has_lstm = lstm_dim > 0
        if lstm_dim > 0:

            self.lstm = LSTM(lstm_dim, time_major=False, dtype='float32', stateful=True, return_sequences=True,
                         return_state=True, name='lstm')

        else :
            self.lstm = None

        self.denses = [Dense(dim, activation=default_activation, dtype="float32", name='dense_%d' % i)
                       for i, dim in enumerate(layer_dims)]

        self.prob = Dense(action_dim, dtype='float32', name="prob", activation="softmax")

    def init_body(self, features):

        if self.has_lstm:
            features, hidden_h, hidden_c = self.lstm(features)
            for layer in self.denses:
                features = layer(features)
            features = self.prob(features)
            return hidden_h, hidden_c
        else:
            for layer in self.denses:
                features = layer(features)
            features = self.prob(features)
            return None, None

    def _compute_feature(self, features):
        if self.has_lstm:
            features, hidden_h, hidden_c = self.lstm(features)
            for layer in self.denses:
                features = layer(features)
            return features, (hidden_h, hidden_c)
        else:
            for layer in self.denses:
                features = layer(features)
            return features, (None, None)

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """

        features, hidden_states = self._compute_feature(states)

        probs = self.prob(features) * (1.0 - self.EPSILON_GREEDY) + self.EPSILON_GREEDY / np.float32(self.action_dim)

        return probs, hidden_states

    @tf.function
    def _get_action_body(self, state):
        probs, hidden_states = self._compute_dist(state)
        action = tf.squeeze(self.dist.sample(probs), axis=1)
        return action, probs, hidden_states

    def __call__(self, state):
        action, probs, (hidden_h, hidden_c) = self._get_action_body(state[np.newaxis][np.newaxis])
        return action.numpy()[0][0], probs.numpy(), hidden_h, hidden_c

    def set_params(self, params):
        for dense, param in zip(self.denses, params['actor_core']):
            dense.set_weights(param)
        self.prob.set_weights(params['actor_head'])
        if self.has_lstm:
            self.lstm.set_weights(params['lstm'])

    def get_params(self):
        actor_weights = [dense.get_weights() for dense in self.denses]
        return {
            'lstm'      : self.lstm.get_weights() if self.has_lstm else None,
            'actor_core': actor_weights,
            'actor_head': self.prob.get_weights(),
        }


class V(tf.keras.Model):
    """
    Value model function
    """

    def __init__(self, layer_dims, default_activation='elu', name='vf'):
        super().__init__(name=name)

        # self.l1 = Dense(128, activation='elu', dtype='float32', name="v_L1")
        #self.l1 = GRU(512, time_major=False, dtype='float32', stateful=True, return_sequences=True)
        self.denses = [Dense(dim, activation=default_activation, dtype='float32') for dim in layer_dims]

        self.v = Dense(1, activation='linear', dtype='float32', name="v")

    def call(self, states):
        features = states
        for layer in self.denses:
            features = layer(features)

        return self.v(features)


class AC(tf.keras.Model, Default):
    def __init__(self, action_dim, layer_dims,
                 lstm_dim):
        super(AC, self).__init__(name='AC')
        Default.__init__(self)

        self.action_dim = action_dim
        self.has_lstm = lstm_dim > 0

        if lstm_dim == 0:
            self.lstm = None
        else:
            self.lstm = LSTM(lstm_dim, time_major=False, dtype='float32', stateful=False, return_sequences=True,
                         return_state=False, name='lstm')
        self.dense_1 = Dense(layer_dims[0], activation='elu', dtype='float32')
        self.V = V(layer_dims[1:])
        self.policy = CategoricalActor(action_dim, self.EPSILON_GREEDY, layer_dims[1:])

        #self.optim = tf.keras.optimizers.RMSprop(rho=0.99, epsilon=1e-5) # Learning rate is affected when training
        self.optim = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-8, clipvalue=4e-3)

        self.step = tf.Variable(0, dtype=tf.int32)

        self.range_ = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(self.TRAJECTORY_LENGTH-1, dtype=tf.int32), axis=0), [self.BATCH_SIZE, 1]),
                                     axis=2)
        self.pattern = tf.expand_dims([tf.fill((self.TRAJECTORY_LENGTH-1,), i) for i in range(self.BATCH_SIZE)], axis=2)

    def train(self, index, parent_index, S, phi, K, l, size,
              training_params, states, actions, rewards, probs, hidden_states, gpu):
        # do some stuff with arrays
        # print(states, actions, rewards, dones)
        # Set both networks with corresponding initial recurrent state
        self.optim.learning_rate.assign(training_params['learning_rate'])

        if training_params['dvd_lambda'] > 0:
            v_loss, mean_entropy, min_entropy, div, min_logp, max_logp, grad_norm \
                = self._train_DvD(S, phi, K, tf.cast(training_params['dvd_lambda'], tf.float32), l, size, parent_index,
                              tf.cast(training_params['entropy_cost'], tf.float32),
                              tf.cast(training_params['gamma'], tf.float32), states, actions, rewards, probs,
                              hidden_states, gpu)
        else:
            v_loss, mean_entropy, min_entropy, div, min_logp, max_logp, grad_norm \
            = self._train(tf.cast(training_params['entropy_cost'], tf.float32),
                          tf.cast(training_params['gamma'],tf.float32), states, actions, rewards, probs, hidden_states, gpu)

        log_name = str(index)
        print(v_loss, div, mean_entropy, grad_norm, tf.reduce_sum(tf.reduce_mean(rewards, axis=0)))

        tf.summary.scalar(name=log_name+"/v_loss", data=v_loss)
        tf.summary.scalar(name=log_name+"/min_entropy", data=min_entropy)
        tf.summary.scalar(name=log_name+"/diversity", data=div)
        tf.summary.scalar(name=log_name+"/mean_entropy", data=mean_entropy)
        tf.summary.scalar(name=log_name+"/ent_scale", data=training_params['entropy_cost'])
        tf.summary.scalar(name=log_name+"/gamma", data=training_params['gamma'])
        tf.summary.scalar(name=log_name+"/learning_rate", data=training_params['learning_rate'])
        tf.summary.scalar(name=log_name+"/min_logp", data=min_logp)
        tf.summary.scalar(name=log_name+"/max_logp", data=max_logp)
        tf.summary.scalar(name=log_name+"/grad_norm", data=grad_norm)
        #tf.summary.scalar(name="misc/distance", data=tf.reduce_mean(states[:, :, -1]))
        tf.summary.scalar(name=log_name+"/reward", data=tf.reduce_sum(tf.reduce_mean(rewards, axis=0)))

        return mean_entropy.numpy()


    @tf.function
    def _train(self, alpha, gamma, states, actions, rewards, probs, hidden_states, gpu):
        '''
        Main training function
        '''
        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):

            actions = tf.cast(actions, dtype=tf.int32)

            with tf.GradientTape() as tape:
                # Optimize the actor and critic
                if self.has_lstm:
                    lstm_states = self.lstm(states, initial_state=[hidden_states[:,0],hidden_states[:, 1]])
                else:
                    lstm_states = states
                lstm_states = self.dense_1(lstm_states)

                v_all = self.V(lstm_states)[: ,:, 0]
                p = self.policy.get_probs(lstm_states[:, :-1])
                kl = tf.divide(p, probs+1e-4)#tf.reduce_sum(p * tf.math.log(tf.divide(p, probs)), axis=-1)
                indices = tf.concat(values=[self.pattern, self.range_, tf.expand_dims(actions, axis=2)], axis=2)
                rho_mu = tf.minimum(1., tf.gather_nd(kl, indices, batch_dims=0))
                targets = self.compute_trace_targets(v_all, rewards, rho_mu, gamma)
                #targets = self.compute_gae(v_all[:, :-1], rewards[:, :-1], v_all[:, -1])
                advantage = tf.stop_gradient(targets) - v_all
                v_loss = tf.reduce_mean(tf.square(advantage))

                p_log = tf.math.log(p + 1e-8)

                ent = - tf.reduce_sum(tf.multiply(p_log, p), -1)
                taken_p_log = tf.gather_nd(p_log, indices, batch_dims=0)



                p_loss = - tf.reduce_mean( tf.stop_gradient(rho_mu) * taken_p_log
                                           * tf.stop_gradient(targets[:, 1:]*gamma + rewards - v_all[:, :-1])
                                           + alpha * ent)


                total_loss = 0.5 * v_loss + p_loss


            grad = tape.gradient(total_loss, self.policy.trainable_variables + self.lstm.trainable_variables
                                 + self.V.trainable_variables + self.dense_1.trainable_variables)

            # x is used to track the gradient size
            x = 0.0
            c = 0.0
            for gg in grad:
                c += 1.0
                x += tf.reduce_mean(tf.abs(gg))
            x /= c

            self.optim.apply_gradients(zip(grad, self.policy.trainable_variables + self.lstm.trainable_variables
                                           + self.V.trainable_variables + self.dense_1.trainable_variables))

            self.step.assign_add(1)
            mean_entropy = tf.reduce_mean(ent)
            min_entropy = tf.reduce_min(ent)
            max_entropy = tf.reduce_max(ent)
            return v_loss, mean_entropy, min_entropy, max_entropy, tf.reduce_min(
                p_log), tf.reduce_max(p_log), x

    @tf.function
    def _train_DvD(self, S, phi, K, lamb, l, size, parent_index,
               alpha, gamma, states, actions, rewards, probs, hidden_states, gpu):
        '''
        Main training function
        '''
        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):

            actions = tf.cast(actions, dtype=tf.int32)

            with tf.GradientTape() as tape:
                # Optimize the actor and critic
                if self.has_lstm:
                    lstm_states = self.lstm(states, initial_state=[hidden_states[:, 0], hidden_states[:, 1]])
                else:
                    lstm_states = states
                lstm_states = self.dense_1(lstm_states)

                v_all = self.V(lstm_states)[:, :, 0]
                p = self.policy.get_probs(lstm_states[:, :-1])
                kl = tf.divide(p, probs + 1e-4)  # tf.reduce_sum(p * tf.math.log(tf.divide(p, probs)), axis=-1)
                indices = tf.concat(values=[self.pattern, self.range_, tf.expand_dims(actions, axis=2)], axis=2)
                rho_mu = tf.minimum(1., tf.gather_nd(kl, indices, batch_dims=0))
                targets = self.compute_trace_targets(v_all, rewards, rho_mu, gamma)
                # targets = self.compute_gae(v_all[:, :-1], rewards[:, :-1], v_all[:, -1])
                advantage = tf.stop_gradient(targets) - v_all
                v_loss = tf.reduce_mean(tf.square(advantage))

                p_log = tf.math.log(p + 1e-8)

                ent = - tf.reduce_sum(tf.multiply(p_log, p), -1)
                taken_p_log = tf.gather_nd(p_log, indices, batch_dims=0)

                p_loss = - tf.reduce_mean(tf.stop_gradient(rho_mu) * taken_p_log
                                          * tf.stop_gradient(targets[:, 1:] * gamma + rewards - v_all[:, :-1])
                                          + alpha * ent)

                behavior_embedding, _ = tf.linalg.normalize(self.policy.get_probs(self.dense_1(self.lstm(S))[:, :-1]))
                new_K = self.compute_kernel(behavior_embedding, phi, K, l, size, parent_index)
                div = tf.linalg.det(new_K + tf.eye(size+1) * 10e-8)

                #behavior_distance = self.compute_distance_score(behavior_embedding, phi, l) + 1e-8

                total_loss = 0.5 * v_loss + p_loss - lamb * tf.math.log(div)

            grad = tape.gradient(total_loss, self.policy.trainable_variables + self.lstm.trainable_variables
                                 + self.V.trainable_variables + self.dense_1.trainable_variables)

            # x is used to track the gradient size
            x = 0.0
            c = 0.0
            for gg in grad:
                c += 1.0
                x += tf.reduce_mean(tf.abs(gg))
            x /= c

            self.optim.apply_gradients(zip(grad, self.policy.trainable_variables + self.lstm.trainable_variables
                                           + self.V.trainable_variables + self.dense_1.trainable_variables))

            self.step.assign_add(1)
            mean_entropy = tf.reduce_mean(ent)
            min_entropy = tf.reduce_min(ent)
            # max_entropy = tf.reduce_max(ent)
            return v_loss, mean_entropy, min_entropy, div, tf.reduce_min(
                p_log), tf.reduce_max(p_log), x

    def compute_kernel(self, new_behavior_embedding, behavior_embeddings, existing_K, l, size, parent_index):

        def similarity_vec(cursor):
            return self.compute_similarity_norm(new_behavior_embedding, behavior_embeddings[cursor], l)

        Kp1 = tf.map_fn(similarity_vec, elems=tf.range((size), dtype=tf.int32), fn_output_signature=tf.float32)

        def similarity(cursor):
            i = cursor // (size+1)
            j = cursor % (size+1)
            if i == j:
                return 1.
            elif i == size:
                return Kp1[j]
            elif j == size:
                return Kp1[i]
            else:
                return existing_K[i, j]

        K = tf.map_fn(similarity, elems=tf.range((size+1)**2, dtype=tf.int32), fn_output_signature=tf.float32)

        return tf.reshape(K, (size+1, size+1))

    def compute_distance_score(self, new_behavior_embedding, pop_embeddings, l):
        return 1.-self.compute_similarity_bc(tf.expand_dims(new_behavior_embedding, 0), pop_embeddings, l)


    def compute_gae(self, v, rewards, last_v, gamma):
        v = tf.transpose(v)
        rewards = tf.transpose(rewards)
        reversed_sequence = [tf.reverse(t, [0]) for t in [v, rewards]]

        def bellman(future, present):
            val, r = present
            return (1. - self.gae_lambda) * val + self.gae_lambda * (
                        r + gamma * future)

        returns = tf.scan(bellman, reversed_sequence, last_v)
        returns = tf.reverse(returns, [0])
        returns = tf.transpose(returns)
        return returns

    def compute_trace_targets(self, v, rewards, rho_mu, gamma):
        # coefs set to 1
        vals_s = tf.transpose(v[:, :-1])
        vals_sp1 = tf.transpose(v[:, 1:])
        last_vr = v[:, -1]# + rewards[:, -1]
        rewards = tf.transpose(rewards) #  rewards[:, :-1]
        rho_mu = tf.transpose(rho_mu)
        reversed_sequence = [tf.reverse(t, [0]) for t in [vals_s, vals_sp1, rewards, rho_mu]]

        def bellman(future, present):
            val_s, val_sp1, r, rm = present

            return val_s+ rm * (r + gamma * val_sp1 - val_s) + gamma * rm \
                   * (future - val_sp1)

        returns = tf.scan(bellman, reversed_sequence, last_vr)
        returns = tf.reverse(returns, [0])
        returns = tf.transpose(returns)
        return tf.concat([returns, tf.expand_dims(last_vr, axis=1)], axis=1)

    @staticmethod
    def compute_similarity_kl(dist_1, dist_2, l):
        return tf.exp(-tf.square(tf.reduce_mean(tf.reduce_sum(
            dist_1 * (tf.math.log(dist_1 + 1e-8) - tf.math.log(dist_2 + 1e-8)), axis=-1)))/(2.*l**2))

    @staticmethod
    def compute_similarity_bc(dist_1, dist_2, l):
        return tf.exp(-tf.square(tf.reduce_mean(-tf.math.log(tf.reduce_sum(tf.sqrt(dist_1*dist_2), axis=-1)+1e-8)))/(2.*l**2))

    @staticmethod
    def compute_similarity_norm(dist_1, dist_2, l):
        return tf.exp(-tf.square(tf.norm(dist_1-dist_2))/(2.*l**2))

    def get_params(self):
        actor_weights = [dense.get_weights() for dense in [self.dense_1] + self.policy.denses]
        return {
            'lstm': self.lstm.get_weights() if self.has_lstm else None,
            'actor_core': actor_weights,
            'actor_head': self.policy.prob.get_weights(),
        }

    def get_training_params(self):
        actor_weights = [dense.get_weights() for dense in self.policy.denses]
        value_weights = [dense.get_weights() for dense in self.V.denses]
        return {
            'lstm'      : self.lstm.get_weights() if self.has_lstm else None,
            'dense_1'   : self.dense_1.get_weights(),
            'actor_core': actor_weights,
            'actor_head': self.policy.prob.get_weights(),
            'value_core': value_weights,
            'value_head': self.V.v.get_weights()
        }

    def set_training_params(self, params):

        if self.has_lstm:
            self.lstm.set_weights(params['lstm'])
        self.dense_1.set_weights(params['dense_1'])
        for dense_layer_weights, dense in zip(params['actor_core'], self.policy.denses):
            dense.set_weights(dense_layer_weights)
        self.policy.prob.set_weights(params['actor_head'])
        for dense_layer_weights, dense in zip(params['value_core'], self.V.denses):
            dense.set_weights(dense_layer_weights)
        self.V.v.set_weights(params['value_head'])

    def get_distribution(self, states):
        with tf.device("/gpu:{}".format(0)):
            x = self.lstm(states)
            x = self.dense_1(x)
            x = self.policy.get_probs(x)
        return x

    def init_body(self, states):
        if self.has_lstm:
            lstm = self.lstm(states)
        else:
            lstm = states
        lstm = self.dense_1(lstm)
        x = self.policy.get_probs(lstm[:, 1:])
        self.V(lstm)
        
    def perturb(self):
        pass
    
    def crossover(self, other_policy):
        parents = [self.get_training_params(), other_policy.get_training_params()]
        l = []
        for w in parents:
            l.append([])
            for layer_name, weights in w.items():
                if 'core' in layer_name:
                    for sub_layer in weights:
                        l[-1].append(sub_layer[0])
                        l[-1][-1] = np.concatenate([l[-1][-1], sub_layer[1][np.newaxis]], axis=0)
                else:
                    if 'lstm' in layer_name:
                        l[-1].append(weights[0][:, :weights[0].shape[1] * 3 // 4])
                        l[-1].append(weights[0][:, weights[0].shape[1] * 3 // 4:])
                        l[-1].append(weights[1])
                        l[-1][-3] = np.concatenate([l[-1][-3], weights[-1][np.newaxis, :weights[0].shape[1] * 3 // 4]],
                                                   axis=0)
                        l[-1][-2] = np.concatenate([l[-1][-2], weights[-1][np.newaxis, weights[0].shape[1] * 3 // 4:]],
                                                   axis=0)
                    else:
                        l[-1].append(weights[0])
                        l[-1][-1] = np.concatenate([l[-1][-1], weights[1][np.newaxis]], axis=0)

        crossovered = nn_crossover(*l, architecture={
            # Automate ?
            0: None, 1: None, 2: None, 3: 1, 4: 3, 5: 4, 6: 3, 7: 6
        })

        count = 0
        for layer_name, weights in parents[0].items():
            if 'core' in layer_name:
                for sub_layer_i in range(len(weights)):
                    weights[sub_layer_i][0][:] = crossovered[count][:-1]
                    weights[sub_layer_i][1][:] = crossovered[count][-1]
                    count += 1
            else:
                if 'lstm' in layer_name:
                    print(weights[0].shape)
                    weights[0][:, :weights[0].shape[1] * 3 // 4] = crossovered[count][:-1]
                    weights[-1][:weights[0].shape[1] * 3 // 4] = crossovered[count][-1]
                    count += 1
                    weights[0][:, weights[0].shape[1] * 3 // 4:] = crossovered[count][:-1]
                    weights[-1][weights[0].shape[1] * 3 // 4:] = crossovered[count][-1]
                    count += 1
                    weights[1][:] = crossovered[count]
                    count += 1

                else:
                    weights[0][:] = crossovered[count][:-1]
                    weights[1][:] = crossovered[count][-1]
                    count += 1

        self.set_training_params(parents[0])