import numpy as np
from cnn import get_cnn
from collections import deque
import random


class DQN:
    def __init__(self,
                 actions,
                 exploration,
                 exploration_decay,
                 exploration_min,
                 gamma,
                 num_batch):
        """
        Initializes the DQN.

        :param environment: Gym Implementation of the game to be trained on
        :param episodes: Number of times the game is repeated from start in
                         order to train
        :param exploration: The probability of exploration
        :param gamma: The delayed depreciation for next state
        :param num_epoch: How many epochs do you want to train the model?
        :param num_batch: The batch size for model during training
        """
        self.exploration = exploration
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.transitions = deque(maxlen=1000)
        self.actions = actions
        self.num_actions = len(actions)
        self.model = get_cnn(self.num_actions)
        self.exploration = exploration
        self.gamma = gamma
        self.batch_size = num_batch

    def get_action(self, state):
        """
        Selects the action with the highest estimated reward with probability
        (1 - \epsilon) approximated by our model. Selects a random action from
        the state space with probability \epsilon.
        :param state: The current state. s_t
        :param env: The environment, used to grab a random action.
        :param epsilon: Probability a random action will be chosen. \epsilon
        :return:
        """
        if np.random.rand() <= self.exploration:
            return random.sample(self.actions, 1)
        return self.predict_best_action(state)

    def store_instance(self, state, action, next_state, reward, is_term):
        self.transitions.append((state, action, next_state, reward, is_term))

    def train(self):
        """

        :return:
        """
        if len(self.transitions) < self.batch_size:
            return

        batch = random.sample(self.transitions, self.batch_size)

        train_x = []
        train_y = []
        for state, action, next_state, reward, is_term in batch:
            train_x.append(state)
            if is_term:
                train_y.append(reward)
            else:
                # If the state is not terminal, move one state forward to
                # compute the terminal state reward
                full_reward = reward + self.gamma * np.max(
                    self.model.predict(DQN.eval_lazy_state(next_state))[0])
                predictions = self.model.predict(DQN.eval_lazy_state(state))
                predictions[0][action] = full_reward
                train_y.append(predictions)

        # Train the model based on train inputs and train labels
        train_x = np.array(train_x)
        train_labels = np.array(train_y).reshape(
            self.batch_size, self.num_actions)
        self.model.fit(train_x, train_labels, batch_size=self.batch_size)
        if self.exploration > self.exploration_min:
            self.exploration *= self.exploration_decay

    def predict_best_action(self, state):
        """
        Predicts the best action for the corrent state for the deep-q network
        :param curr_state: The current preprocessed state of the game
        :return:
        """
        return self.actions[np.argmax(
            self.model.predict(DQN.eval_lazy_state(state))[0])]

    def save(self):
        self.model.save_weights('my_model_weights.h5')

    def load(self):
        self.model.load_weights('my_model_weights.h5')

    @staticmethod
    def eval_lazy_state(state):
        return np.expand_dims(np.array(state), 0)

