
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import DAE_params as prms
from time import time
import datetime

#################################################
# 
# This script activates each node at the 
# bottleneck layer for the full ensemble 
# of networks and computes the resulting
# changes in predicted gene expression.
#
#################################################

#################################################
# Import all network and training parameters
#################################################

data_collection = prms.data_collection
tied = prms.tied
layer_actis = prms.layer_actis
layer_sizes = prms.layer_sizes
layer_dprts = prms.layer_dprts
learning_rate = prms.lr
epochs = prms.epcs
batch_size = prms.bs
corr_lvl = prms.corr_lvl

bn_layer = min(layer_sizes)
num_layers = len(layer_sizes)

compute_BN_activations = False	# Set to True if the conditions which activate each node are desired to be known
num_jobs = 10			# The number of networks to examine
ens_index = 0			# If multiple ensembles have been run, this index can be changed to distinguish between them 
jobs = range(num_jobs)

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

################################################
# Import training and test data.
################################################

print('Importing data set... ')
t0 = time()
infoFilePath = 'Data/' + data_collection + "_0_formatted.ftr" # Only this single data split is used for bottleneck node activation
readFrame = pd.read_feather(infoFilePath, columns=None, use_threads=True)
print('Complete! ', datetime.timedelta(seconds=time() - t0))

x_train = []
x_val = []
x_test = []

labels_train = []
labels_val = []
labels_test = []

genes = list(readFrame['genes'])
num_genes = len(genes)
for data_set in list(readFrame):
	if data_set[:5] == 'train':
		x_train.append(readFrame[data_set])
		labels_train.append(data_set[6:])
	elif data_set[:3] == 'val':
		x_val.append(readFrame[data_set])
		labels_val.append(data_set[4:])
	elif data_set[:4] == 'test':
		x_test.append(readFrame[data_set])
		labels_test.append(data_set[5:])

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)

print('Data set: '+data_collection)
print()
print('\tGenes: ', num_genes)
print('\tTraining: ', x_train.shape)
print('\tValidation: ', x_val.shape)
print('\tTest: ', x_test.shape)
print()

################################################
# Import networks and compute bottleneck 
# activations and predicted expression vectors
# for each bottleneck node.
################################################

bn_expr = []
BN_acti_data = {}

for job in jobs:
			
	file_name = data_collection + '_' +\
			'tied_' + str(tied) +\
			'_layers_' + '_'.join([str(layer) for layer in layer_sizes]) +\
			'_activs_' + '_'.join(layer_actis) +\
			'_lr_' + str(learning_rate) +\
			'_epcs_' + str(epochs) +\
			'_bs_' + str(batch_size) +\
			'_dprs_' + '_'.join([str(layer_dprt) for layer_dprt in layer_dprts]) + '_' + str(job) + '.ftr' 
		
	
	################################################
	# Open network file and assemble network 
	# components into a dictionary.
	################################################
	
	print('Importing network', file_name)
	print()
	t0 = time()
	infoFilePath = 'Networks/'+file_name
	try:
		readFrame = pd.read_feather(infoFilePath, columns=None, use_threads=True)
	
	except FileNotFoundError as err:
		print(infoFilePath, err)
	
	network = {}
	
	for compo in list(readFrame):
		#💀️
		if compo[0] == 'w':
			W = []
			for term in readFrame[compo]:
				try:
					if len(term) > 1:
						W.append(term)
				except TypeError as err:
					err
					
			print('\t', compo, np.shape(W))
			network[compo] = np.array(W)
			
		if compo[0] == 'b':
			b = []
			for term in readFrame[compo]:
				if not np.isnan(term):
					b.append(term)
			print('\t', compo, np.shape(b))
			network[compo] = np.array(b)
			
	
	if tied:
		for i in range(num_layers):
			network['w' + str(2*num_layers - i)] = np.transpose(network['w' + str(i + 1)])
	print()
	print('Complete! ', datetime.timedelta(seconds=time() - t0))
	print()
	
	################################################
	# Compute predicted expression vectors for 
	# each bottleneck node.
	################################################
	
	print('Activating bottleneck nodes...')
	t0 = time()
	BN_vecs = []
	
	for i in range(bn_layer):
		A = np.zeros(bn_layer)
		for j in range(num_layers + 1, 2*num_layers + 1):
			W = network['w'+str(j)]
			b = network['b'+str(j)]
			A = sigmoid(np.dot(A, W) + b)
		
		Ai = np.array(A)
		
		A = np.array([int(j == i) for j in range(bn_layer)])
		for j in range(num_layers + 1, 2*num_layers + 1):
			W = network['w'+str(j)]
			b = network['b'+str(j)]
			A = sigmoid(np.dot(A, W) + b)
			
		BN_vecs.append(str(job) + '_' + str(i))
		bn_expr.append(list(A - Ai))
	print('Complete! ', datetime.timedelta(seconds=time() - t0))
	print()
	
	if compute_BN_activations:
		################################################
		# Compute predicted expression vectors for 
		# each bottleneck node.
		################################################
		
		print('Determining bottleneck node activations...')
		t0 = time()
		acti_mat = []
		acti_labels = []
		print()
		print('\tTraining data')
		for i,x in enumerate(x_train):
			A = np.array(x)
			label = labels_train[i]
			for j in range(1, num_layers + 1):
				W = network['w'+str(j)]
				b = network['b'+str(j)]
				A = sigmoid(np.dot(A, W) + b)
			
			acti_labels.append(label)
			acti_mat.append(list(A))
		
		print('\tValidation data')
		for i,x in enumerate(x_val):
			A = np.array(x)
			label = labels_val[i]
			for j in range(1, num_layers + 1):
				W = network['w'+str(j)]
				b = network['b'+str(j)]
				A = sigmoid(np.dot(A, W) + b)
			
			acti_labels.append(label)
			acti_mat.append(list(A))
		
		print('\tTest data')
		for i,x in enumerate(x_test):
			A = np.array(x)
			label = labels_test[i]
			for j in range(1, num_layers + 1):
				W = network['w'+str(j)]
				b = network['b'+str(j)]
				A = sigmoid(np.dot(A, W) + b)
			
			acti_labels.append(label)
			acti_mat.append(list(A))
		
		for i,BN_acti in enumerate(np.transpose(acti_mat)):
			BN_acti_data[BN_vecs[i]] = BN_acti
		print()
		print('Complete! ', datetime.timedelta(seconds=time() - t0))
		print()

################################################
# Convert results to a dictionary for explort.
################################################
BN_activ_data = {"genes":genes}
for i,net_BN in enumerate(BN_vecs):
	BN_activ_data[net_BN] = list(bn_expr[i])
	
##############################################
# Write the BN vector information to a file
##############################################
print('Writing results to .ftr files...')
t0 = time()
file_name = data_collection + '_' +\
		'tied_' + str(tied) +\
		'_layers_' + '_'.join([str(layer) for layer in layer_sizes]) +\
		'_activs_' + '_'.join(layer_actis) +\
		'_lr_' + str(learning_rate) +\
		'_epcs_' + str(epochs) +\
		'_bs_' + str(batch_size) +\
		'_jobs_' + str(num_jobs) +\
		'_dprs_' + '_'.join([str(layer_dprt) for layer_dprt in layer_dprts]) + '_' + str(ens_index)

if compute_BN_activations:
	BN_acti_data['labels'] = acti_labels
	infoFilePath = 'BN_activations/' + file_name + "_BN_datasets.ftr"
	dataFrame = pd.DataFrame(data=BN_acti_data)
	dataFrame.to_feather(infoFilePath)

infoFilePath = 'BN_activations/' + file_name + "_BN_activations.ftr"
dataFrame = pd.DataFrame(data=BN_activ_data)
dataFrame.to_feather(infoFilePath)

print('Complete! ', datetime.timedelta(seconds=time() - t0))


	
