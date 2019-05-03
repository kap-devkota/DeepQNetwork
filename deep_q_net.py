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
        Initializes a Deep Q network.

        :param num_actions: The number of actions that are possible in the
        environment this Deep Q network will be working with.
        :param exploration: The exploration rate.
        :param exploration_decay: The decay rate of the exploration rate.
        :param exploration_min: The minimum possible exploration rate.
        :param reward_decay: The rate at which reward decays at each time step.
        :param deque_size: The size of the deque that will hold past experiences
        for replay.
        :param batch_size: The batch size to use when training the NN.
        :param model: The NN to use to approximate the Q function.
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
        Updates the exploration probability. Should only be called after
        training.

        :return: None.
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
        :return: An action.
        """

        if np.random.rand() <= self.exploration and is_train:
            return random.sample(self.actions, 1)[0]
        return self.predict_best_action(state)

    def store(self, info):
        """
        Stores past experiences in the replay buffer.
        :param info: A list of tuples consisting of state, action, next_state,
        reward, is_term to add to the replay buffer.
        :return: None.
        """

        self.transitions.extend(info)
        
    def train(self, num_samples_scale=10, epochs=1):
        """
        Trains the NN on past experiences stored in the replay buffer.

        :param num_samples_scale: Determines the number of batches that need
        to be in the replay buffer before training begins.
        :param epochs: The number of epochs to train for.
        :return: None.
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
        train_y = np.array(train_y).reshape(self.batch_size, self.num_actions)
        self.model.fit(
            train_x, train_y, batch_size=self.batch_size, epochs=epochs)
        self.update_exploration()

    def predict_best_action(self, state):
        """
        Predicts the best action for the current state for the deep-q network
        :param state: The current preprocessed state of the game
        :return: The best action according to the NN.
        """

        l = self.model.predict(DQN.eval_lazy_state(state))[0]
        l = np.argmax(l)
        return self.actions[l]

    def save(self, suffix):
        """
        Saves the model weights to disk.
        :param suffix: The suffix to add to the filename.
        :return: None.
        """

        self.model.save_weights('weights_{}.h5'.format(suffix))

    def load(self, suffix):
        """
        Loads the model weights from disk.
        :param suffix: The suffix to add to the filename.
        :return: None
        """

        self.model.load_weights('weights_{}.h5'.format(suffix))

    @staticmethod
    def eval_lazy_state(state):
        """
        Evaluate the lazy state representation. Currently each frame is stored
        only once even though that frame should reside within 4 different
        states. This turns thats that lazy representation of a state into
        an actual copy.

        :param state: The state to transform.
        :return: The actual state with 4 real frames.
        """

        return np.expand_dims(np.array(state), 0)

