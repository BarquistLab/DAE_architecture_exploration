
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams['svg.fonttype'] = 'none' 
import matplotlib as mpl
mpl.rc('font',family='times new roman')
import numpy as np
import pandas as pd
import DAE_params as prms


#################################################
# 
# This script finds all training trajectories
# corresponding to the training and network 
# parameters specified in DAE_params.py.
# It then generates a figure of these trajectories
# like figure S4 in the supplemental text, and
# determines the optimal number of epochs to 
# employ early stopping.
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

jobs = [i for i in range(100)]# The number of training trajectories to search for

ave_loss_hist = np.zeros(epochs)
ave_val_loss_hist = np.zeros(epochs)
ave_test_loss = 0

ave2_loss_hist = np.zeros(epochs)
ave2_val_loss_hist = np.zeros(epochs)
ave2_test_loss = 0

file_name = data_collection + '_' +\
		 'tied_' + str(tied) +\
		'_layers_' + '_'.join([str(layer) for layer in layer_sizes]) +\
		'_activs_' + '_'.join(layer_actis) +\
		'_lr_' + str(learning_rate) +\
		'_epcs_' + str(epochs) +\
		'_bs_' + str(batch_size) +\
		'_cl_' + str(corr_lvl) +\
		'_dprs_' + '_'.join([str(layer_dprt) for layer_dprt in layer_dprts])

print(file_name) # file_name specifies the network and training parameters corresponding to the trajectories of interest

epcs_list = [i for i in range(1, epochs + 1)]

plt.figure(figsize = (7,4))
ax1 = plt.gca()
ax2 = ax1.twinx()
jobs_found = 0

for job in jobs:
	
	#################################################
	# Search for all trained network trajectories 
	# representing networks with the parameters
	# as speciried in DAE_params.py
	#################################################
	
	file_found = True
	
	try:
		infoFilePath = 'Training_trajs/'+file_name+'_' + str(job) + '.ftr'
		readFrame = pd.read_feather(infoFilePath, columns=None, use_threads=True)
		
	except FileNotFoundError as err:
		file_found = False
	
	if file_found:
		
		#################################################
		# Compute average losses and squared losses
		# and plot each trajectory.
		#################################################
		jobs_found += 1
		
		loss_hist = readFrame['loss']
		val_loss_hist = readFrame['val_loss']
		test_loss = readFrame['test_loss'][0]
		
		ave_loss_hist = ave_loss_hist + np.array(loss_hist)
		ave_val_loss_hist = ave_val_loss_hist + np.array(val_loss_hist)
		ave_test_loss = ave_test_loss + test_loss
		
		ave2_loss_hist = ave2_loss_hist + np.array(loss_hist)**2
		ave2_val_loss_hist = ave2_val_loss_hist + np.array(val_loss_hist)**2
		ave2_test_loss = ave2_test_loss + test_loss**2
		
		handle1, = ax1.plot(epcs_list, loss_hist, color = 'blue', alpha = .3, label = 'Loss')
		handle2, = ax2.plot(epcs_list, val_loss_hist, color = 'red', alpha = .3, label = 'Val. loss')
		handle3, = ax2.plot([epcs_list[-1]], [test_loss], 'go', alpha = .3, label = 'Test loss')


#################################################
# Find the optimal training epochs based on 
# the minimum of the average validation
# loss, then find the average and standard
# deviation of the training and test losses
# at this point. The average and standard
# deviation of the test loss is only computed
# at the end of the full trajectories.
#################################################

ave_test_loss = ave_test_loss/jobs_found
ave2_test_loss = ave2_test_loss/jobs_found
ave_val_loss_hist = ave_val_loss_hist/jobs_found
ave2_val_loss_hist = ave2_val_loss_hist/jobs_found
ave_loss_hist = ave_loss_hist/jobs_found
ave2_loss_hist = ave2_loss_hist/jobs_found

ave_val_loss = min(ave_val_loss_hist)
opt_epchs = np.where(ave_val_loss_hist == ave_val_loss)[0][0]
ave2_val_loss = ave2_val_loss_hist[opt_epchs]

ave_loss = ave_loss_hist[opt_epchs]
ave2_loss = ave2_loss_hist[opt_epchs]

std_loss = np.sqrt(ave2_loss - ave_loss**2)*np.sqrt(len(jobs)/(len(jobs)-1))
std_val_loss = np.sqrt(ave2_val_loss - ave_val_loss**2)*np.sqrt(len(jobs)/(len(jobs)-1))
std_test_loss = np.sqrt(ave2_test_loss - ave_test_loss**2)*np.sqrt(len(jobs)/(len(jobs)-1))

print()
print('Training parameters:')
print()
print('\tlearning rate =', learning_rate)
print('\tbatch size =', batch_size)
print('\tepochs =', epochs)
print()
print('Losses at optimum:')
print()
print('\ttraining loss =', round(ave_loss, 4), '+/-', round(std_loss, 4))
print('\tvalidation loss =', round(ave_val_loss, 4), '+/-', round(std_val_loss, 4))
print('\ttest loss =', round(ave_test_loss, 4), '+/-', round(std_test_loss, 4))
print()
print('Optimal # epochs =', opt_epchs + 1)

#################################################
# Finish and save the training trajectory figure.
#################################################

handle4, = ax1.plot(epcs_list, ave_loss_hist, lw=1.25, color = 'cyan', linestyle = 'solid', label = 'Ave. loss')
handle5, = ax2.plot(epcs_list, ave_val_loss_hist, lw=1.25, color = 'yellow', linestyle = 'solid', label = 'Ave. val. loss')
handle6, = ax2.plot(epcs_list[-1], ave_test_loss, 'mo', lw=1.25, label = 'Ave. test loss')
handle7 = plt.axvline(opt_epchs + 1, 0, 1, lw=1.25, linestyle='dashed', color = 'grey', label = 'Opt. epochs')
plt.legend(handles = [handle1, handle2, handle3, handle4, handle5, handle6, handle7], fontsize = 8, bbox_to_anchor=(1.45, 1.05))
plt.title('Learning rate = '+str(learning_rate)+', batch size = '+str(batch_size), fontsize = 12)
ax1.set_xlabel('Epochs', fontsize = 12)
ax1.set_ylabel('Training loss', fontsize = 12)
ax2.set_ylabel('Validation / test loss', fontsize = 12)
plt.tight_layout()
plt.savefig('Figures/' + file_name + '_traj.svg', format = 'svg', dpi=300)
plt.clf()
				
		
			
