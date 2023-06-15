data_collection = "E_coli_PRECISE2"				# Data set to train on.

tied = True						# Specifies if the autoencoder is tied.
layer_sizes = [2000, 1000, 50]				# Sizes of hidden layers up to and including the bottleneck. If untied, decoder layers must be given explicitly.
layer_actis = ['sigmoid' for i in range(2*len(layer_sizes))]	# Activation functions for each layer.
layer_dprts = [0.0 for i in range(2*len(layer_sizes))]		# Optional dropout rates for each layer.
save_net = True						# Set to False if only the training trajectory is needed, and not the full network.

corr_lvl = .1						# Input data corruption level to train denoising.
lr = 1e-4							# Training learning rate.
epcs = 154						# Number of epochs to train.
bs = 5							# Batch size for training.


