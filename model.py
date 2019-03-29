from tensorflow import keras

#Model to be defined in the Model class

class Model:
	def __init__(self):
		pass

	def train(self , train_input , train_label , num_epoch , num_batch , loss_func = "RMSE"):
		"""
		Trains the model for given input and label
		@params:
			train_input: List of inputs to train
			train_label: List of labels
			num_epoch: Number of epoch to train
			num_batch: Batch Size
			loss_func: The loss function to be employed by the model
		"""
		pass

	def predict(self , ip):
		"""
		Given an input, finds the prediction value
		@params:
			ip: List of inputs
		@returns:
			pred: List of predictions
		"""
		pred = None


		return pred
