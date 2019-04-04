import state
import random
import numpy as np
from cnn import get_cnn


class DQN:
    def __init__(self,
                 environment,
                 episodes,
                 num_exp,
                 exploration,
                 gamma,
                 num_epoch,
                 num_batch):
        """
        Initializes the DQN.

        :param environment: Gym Implementation of the game to be trained on
        :param episodes: Number of times the game is repeated from start in
                         order to train
        :param num_exp: The size of recall list
        :param exploration: The probability of exploration
        :param gamma: The delayed depreciation for next state
        :param num_epoch: How many epochs do you want to train the model?
        :param num_batch: The batch size for model during training
        """
        self.model = get_cnn(environment.action_space.n)
        self.environment = environment
        # List of integers that represents all the actions possible in the game
        self.actions = range(environment.action_space.n)
        self.episodes = episodes
        self.num_exp = num_exp
        self.exploration = exploration
        self.gamma = gamma
        self.num_epoch = num_epoch
        self.num_batch = num_batch

    # epsilon and decay

    def train(self):
        for i in range(self.episodes):

            _state = state.preprocess_pong_img(
                self.environment.reset())

            # Initialize recall states
            transition_sets = []

            for j in range(self.num_exp):

                # Random checking for exploration
                is_explore = np.random.binomial(1, self.exploration)

                # Choose action based on the exploration
                if is_explore:
                    action = self.environment.action_space.sample()
                else:
                    # The predict function in model gives the reward, given a
                    # state and an action. This gives the most rewarding action

                    best_act = np.argmax(
                        self.model.predict(_state)[0])
                    action = self.actions[best_act]

                # Apply this action to the environment, get the next state and
                # reward, preprocess the next state
                _n_state, reward, is_term, _ = self.environment.step(action)
                _n_state = state.preprocess_pong_img(_n_state)

                # Save all to transition_sets
                transition_sets.append((
                    _state, action, _n_state, reward, is_term))

                if is_term:
                    break

                # Change to next state
                _state = _n_state

            if len(transition_sets) >= self.num_batch:
                # Training the model based on the recall states

                batch = random.sample(transition_sets, self.num_batch)

                # Label is the prospective rewards
                train_labels = []
                train_x = []

                for init_st, act, next_st, reward, is_term in batch:
                    train_x.append(init_st)
                    if is_term:
                        train_labels.append(reward)
                    else:
                        # If the state is not terminal, move one state forward to
                        # compute the terminal state reward
                        full_reward = reward + self.gamma * np.max(
                            self.model.predict(next_st)[0])
                        pred = self.model.predict(init_st)
                        pred[0][act] = full_reward
                        train_labels.append(pred)

                # Train the model based on train inputs and train labels
                train_x = np.squeeze(np.array(train_x), 1)
                train_labels = np.array(train_labels).reshape(10, 6)
                self.model.fit(train_x,
                               train_labels,
                               batch_size=self.num_batch)

# def predict_best_action(self , curr_state):
# 	"""
# 	Predicts the best action for the corrent state for the deep-q network
# 		@param:
# 			curr_state: The current preprocessed state of the game
# 	"""
# 	return self.actions[np.argmax([
# 		self.model.predict(state.compress(curr_state , act))
# 		for act in self.actions)])]
#
