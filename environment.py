#Gym related commands to be defined here

class Environment:
	def __init__(self):
		pass

	def apply(self , action):
		"""
		Given an action, changes the environment based on the action and returns reward and the next state
		@params:
			action: A value that represents an action to the environment
		
		@returns:
			n_state: The next state of the environment
			reward: The reward gained during transition		
		"""	

		n_state = None
		reward = None
	
		return n_state , reward

	def get_initial_state(self):
		"""
		Returns the initial state
		@returns:
			initial_state: The initial state of the environment
		"""
		initial_state = None

		return initial_state
		
