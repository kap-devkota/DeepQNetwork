import numpy as np
from cnn import get_cnn
from collections import deque
import random


class DQN:
    def __init__(self,
                 num_actions,
                 exploration,
                 exploration_decay,
                 exploration_min,
                 reward_decay,
                 deque_size,
                 batch_size,
                 model=None):
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

        self.exploration_min = exploration_min
        self.exploration = exploration
        self.exploration_decay = exploration_decay

        self.transitions = deque(maxlen=deque_size)
        
        self.actions = list(range(num_actions))
        self.num_actions = num_actions
        self.exploration = exploration
        
        self.reward_decay = reward_decay

        self.batch_size = batch_size

        if model is None:
            self.model = get_cnn(self.num_actions)

        return

    def update_exploration(self):
        """
        Update the exploration probability if the number of iterations reaches a point
        :return:
        """
        if self.exploration <= self.exploration_min:
            return

        self.exploration *= self.exploration_decay

        return
    
    def get_action(self, state, is_train=True):
        """
        Selects the action with the highest estimated reward with probability
        (1 - \epsilon) approximated by our model. Selects a random action from
        the state space with probability \epsilon.
        :param state: The current state. s_t
        :param is_train: Checks if the get_action mode is train or test
        :return:
        """
        if np.random.rand() <= self.exploration and is_train:
            return random.sample(self.actions, 1)[0]
        return self.predict_best_action(state)

    def store(self, info):
        """
        Store the 5-tuple representing state transitions into the deque
        :param state: (84 x 84 x 4) image containing the current state
        :param action: An integer representing the action taken during the state
        :param next_state: (84 x 84 x 4) image representing the next state after the action
                        is applied
        :param reward: A float representing the reward
        :param is_term: A boolean value representing if the state is terminal
        """
        self.transitions.extend(info)
        
    def train(self, num_samples_scale=10, epochs=1):
        """
        Trains the DQN model from the samples collected in the deque
        :param num_samples_scale: The ratio of the sample to the batch size

        :return: NoneType
        """
        num_samples = self.batch_size * num_samples_scale
        
        if len(self.transitions) < num_samples:
            return
        
        batch = random.sample(self.transitions, self.batch_size)

        train_x = []
        train_y = []
        for state, action, next_state, reward, is_term in batch:
            train_x.append(state)
            # Find the predicted reward for all action using the deep model
            predictions = self.model.predict(DQN.eval_lazy_state(state))

            if is_term:
                predictions[0][action] = reward
                train_y.append(predictions)

            else:
                # Find the action that yields the largest reward in the next state
                # and find the full_reward label for that action
                next_state_pred = self.model.predict(
                    DQN.eval_lazy_state(next_state))[0]
                pred_reward = reward + \
                              self.reward_decay * \
                              np.max(next_state_pred)
                predictions[0][action] = pred_reward
                train_y.append(predictions)

        # Train the model based on train inputs and train labels
        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(num_samples, self.num_actions)
        self.model.fit(
            train_x, train_y, batch_size=self.batch_size, epochs=epochs)
        self.update_exploration()

    def predict_best_action(self, state):
        """
        Predicts the best action for the current state for the deep-q network
        :param state: The current preprocessed state of the game
        :return:
        """
        l = self.model.predict(DQN.eval_lazy_state(state))[0]
        l = np.argmax(l)
        return self.actions[l]

    def save(self):
        self.model.save_weights('my_model_weights.h5')

    def load(self):
        self.model.load_weights('my_model_weights.h5')

    @staticmethod
    def eval_lazy_state(state):
        return np.expand_dims(np.array(state), 0)

