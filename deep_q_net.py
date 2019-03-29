import state
from model import Model
from environment import Environment 
import numpy as np

class DQN:
	def __init__(self , model , environment , episodes , num_exp , exploration , gamma , num_epoch , num_batch):
		"""
		@params:
			model: Deep Model to compute the reward of given state-action combination
			environment: Gym Implementation of the game to be trained on
			episodes: Number of times the game is repeated from start in order to train
			num_exp: The size of recall list
			exploration: The probability of exploration
			gamma: The delayed depreciation for next state
			num_epoch: How many epochs do you want to train the model?
			num_batch: The batch size for model during training
		"""
		self.model = model
		self.environment = environment
		self.actions = state.actions	#List of integers that represents all the actions possible in the game
		self.episodes = episodes
		self.num_exp = num_exp
		self.exploration = exploration
		self.gamma = gamma
		self.num_epoch = num_epoch
		self.num_batch = num_batch
	

	def train(self):				
		for i in range(self.episodes):

			_state = state.preprocess(self.environment.get_initial_state())

			#Initialize recall states		
			transition_sets = []					
			
			for j in range(self.num_exp):	
				
				#Random checking for exploration		
				is_explore = np.random.binomial(1 , self.exploration)
	
				#Choose action based on the exploration
				if(is_explore == True):
					action = self.actions[int(np.random.uniform(0 , len(self.actions)))]
				
				else:
					#The predict function in model gives the reward, given a state and an action
					#This gives the most rewarding action
					action = self.actions[np.argmax([self.model.predict(state.compress(_state , act)) for act in self.actions])]							
				
				#Apply this action to the environment, get the next state and reward, preprocess the 
				#next state
				_n_state, reward = self.environment.apply(action)
				_n_state = state.preprocess(_n_state)

				#Save all to transition_sets
				transition_sets.append((_state , action , _n_state , reward , state.is_treminal(reward)))
				
				if(state.is_terminal(reward)):
					break
				
				#Change to next state
				_state = _n_state
			
			#Training the model based on the recall states
			
			#Input is the combination of state and action
			train_inputs = [state.compress(s , a) for (s , a , n , r , i) in transition_sets]

			#Label is the prospective rewards			
			train_labels = []
			
			for t_set in transition_sets:
				init_st , act , next_st , reward , is_term = t_set
				if(is_term == True):
					train_labels.append(reward)
				else:
					#If the state is not terminal, move one state forward to compute the terminal state reward
					delayed_reward = self.gamma * np.max([self.model.predict(state.compress(next_st ,act))
									     for act in self.actions])							
					train_labels.append(reward + delayed_reward)


			#Train the model based on train inputs and train labels
			self.model.train(train_inputs , train_labels , num_epoch = self.num_epoch , num_batch = self.num_batch , loss_func = "RMSE")
			
	def predict_best_action(self , curr_state):
		"""
		Predicts the best action for the corrent state for the deep-q network
			@param:
				curr_state: The current preprocessed state of the game
		"""
		return self.actions[np.argmax([self.model.predict(state.compress(curr_state , act)) for act in self.actions)])]

