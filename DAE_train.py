import numpy as np
from numpy.random import default_rng
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)# Limit number of threads
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout
from tensorflow.keras.initializers import RandomUniform
import pandas as pd
from time import time
import datetime
import sys
import DAE_params as prms
import os

################################################
# 
# This script constructs and trains a single
# denoising autoencoder on the data set
# with random training / validation / test
# split specified by the user-provided
# index (net_index).
#
################################################

################################################
# Get the training and network parameters
################################################

data_collection = prms.data_collection
tied = prms.tied
layer_actis = prms.layer_actis
layer_sizes = prms.layer_sizes
num_layers = len(layer_sizes)
bn_layer = min(layer_sizes)
corr_lvl = prms.corr_lvl
save_net = prms.save_net
layer_dprts = prms.layer_dprts

net_index = int(sys.argv[1])
learning_rate = prms.lr
epochs = prms.epcs
batch_size = prms.bs

file_path = 'Data/'
file_name = data_collection + '_' + 'tied_' + str(tied) +\
		'_layers_' + '_'.join([str(layer) for layer in layer_sizes]) +\
		'_activs_' + '_'.join(layer_actis) + '_lr_' + str(learning_rate) +\
		'_epcs_' + str(epochs) + '_bs_' + str(batch_size) +\
		'_dprs_' + '_'.join([str(layer_dprt) for layer_dprt in layer_dprts]) + '_' + str(net_index) + '.ftr'

if os.path.isfile(file_path + 'Networks/'+file_name):
	print('ABORTED: Run already completed.')
	exit()

###############################################
# Define NN layer transpose so that 
# DAEs can be tied.
###############################################

class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", 
                                     shape=[self.dense.input_shape[-1]],
                                     initializer="zeros")
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

################################################
# Import training, validation and test data.
################################################

print('Importing data set... ')
t0 = time()
infoFilePath = file_path + data_collection + "_" + str(net_index%100) + "_formatted.ftr"
readFrame = pd.read_feather(infoFilePath, columns=None, use_threads=True)
print('Complete! ', datetime.timedelta(seconds=time() - t0))

x_train = []
x_val = []
x_test = []

genes = list(readFrame['genes'])
num_genes = len(genes)
for data_set in list(readFrame):
	if data_set[:5] == 'train':
		x_train.append(list(readFrame[data_set]))
	elif data_set[:3] == 'val':
		x_val.append(list(readFrame[data_set]))
	elif data_set[:4] == 'test':
		x_test.append(list(readFrame[data_set]))

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)

print('Data set: '+data_collection + "_" + str(net_index%100))
print()
print('\tGenes: ', num_genes)
print('\tTraining: ', x_train.shape)
print('\tValidation: ', x_val.shape)
print('\tTest: ', x_test.shape)
print()

################################################
# Create a corrupted (noised) data set with 
# randomly assigned zeroes such to 
# train the autoencoder to denoise  
# (-> denoising autoencoder) and improve model 
# generalizability. 
################################################

corr_genes = int(corr_lvl*num_genes)
print('Corruption level: '+str(corr_lvl)+', '+str(corr_genes)+' genes')
print()
rng = default_rng()
corr_x_train = []
corr_x_val = []
corr_x_test = []

for data_set in x_train:
	new_data_set = np.array(data_set)
	new_data_set[rng.choice(range(num_genes), size=corr_genes, replace=False)] = 0
	corr_x_train.append(list(new_data_set))

for data_set in x_val:
	new_data_set = np.array(data_set)
	new_data_set[rng.choice(range(num_genes), size=corr_genes, replace=False)] = 0
	corr_x_val.append(list(new_data_set))

for data_set in x_test:
	new_data_set = np.array(data_set)
	new_data_set[rng.choice(range(num_genes), size=corr_genes, replace=False)] = 0
	corr_x_test.append(list(new_data_set))

corr_x_train = np.array(corr_x_train)
corr_x_val = np.array(corr_x_val)
corr_x_test = np.array(corr_x_test)

################################################
# Build a DAE based on the specified 
# model parameters.
################################################

layers = []
dropouts = []
inputs = keras.Input(shape=(num_genes))
x = Flatten()(inputs)
print('Input size: ', num_genes)
for i,layer_size in enumerate(layer_sizes):
	dropout_i = Dropout(rate=layer_dprts[i], input_shape=(layer_size,))
	dense_i = Dense(layer_size, activation=layer_actis[i], kernel_initializer=RandomUniform(minval=-.1, maxval=.1, seed=None))	
	dropouts.append(dropout_i)
	layers.append(dense_i)
	x = dropout_i(x)
	x = dense_i(x)
	print('\tLayer '+str(i)+': ..>'+str(layer_size)+', '+layer_actis[i])


if tied:
	for i in range(num_layers):
		j = num_layers - i - 1
		layer = layers[j]
		x = Dropout(rate=layer_dprts[num_layers + i], input_shape=(layer_sizes[j],))(x)
		x = DenseTranspose(layer, activation=layer_actis[num_layers + i])(x)
		print('\tLayer '+str(i + num_layers)+': '+str(layer_sizes[j])+'>.., '+layer_actis[num_layers + i])
		

outputs = Reshape([num_genes])(x)
ae = keras.Model(inputs=inputs, outputs=outputs)
print()
################################################
# Initialize the optimizer and train the DAE.
################################################
print('Begin training...')
opt = keras.optimizers.Adam(learning_rate=learning_rate)
ae.compile(loss='binary_crossentropy', optimizer=opt)

time_i = time()
history = ae.fit(corr_x_train, x_train,
			epochs=epochs,
			batch_size=batch_size,
			validation_data=(corr_x_val, x_val), verbose=1, shuffle=True)
time_f = time()
print('Training time = ', datetime.timedelta(seconds=time() - t0))
test_loss = ae.evaluate(corr_x_test, x_test)
############################################
# Save the network as a .ftr file.
############################################
if save_net:
	wm_index = 0
	bv_index = 0

	data = {}
	print('Writing network file...')
	t0 = time()
	infoFilePath = 'Networks/' + file_name

	data['l, vl, tl, t, o'] = [history.history['loss'][-1], history.history['val_loss'][-1], test_loss, time_f - time_i]

	df_list = [pd.DataFrame(data)]

	for entry in ae.get_weights():
		print(np.shape(entry))
		if len(np.shape(entry)) == 2:
			wm_index += 1
			entry_name = 'w'+str(wm_index)
		else:
			bv_index += 1
			entry_name = 'b'+str(bv_index)
		
		df_list.append(pd.DataFrame({entry_name:list(entry)}))

	dataFrame = pd.concat(df_list, axis=1)
	dataFrame.to_feather(infoFilePath)
	print('DAE network saved. ', datetime.timedelta(seconds=time() - t0))

############################################
# Write loss / val_loss histories
# and final test score to output file.
############################################
print('Saving training loss trajectories... ')
t0 = time()
data = {}

infoFilePath = 'Training_trajs/' + file_name

data['loss'] = history.history['loss']
data['val_loss'] = history.history['val_loss']
data['test_loss'] = test_loss

dataFrame = pd.DataFrame(data=data)
dataFrame.to_feather(infoFilePath)
print('Complete. ', datetime.timedelta(seconds=time() - t0))

