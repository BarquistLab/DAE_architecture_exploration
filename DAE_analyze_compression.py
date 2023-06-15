
import numpy as np
from scipy.stats import hypergeom
import pandas as pd
from time import time
import datetime
import sys

def list_format(entry):
	return entry.replace("['", '').replace("']", '').split("', '")

store_probs = False # Set to True to generate a file with computed probabilities to make future analyese faster.
data_collection = "E_coli_PRECISE2"
file_name = 'E_coli_PRECISE2_tied_True_layers_2000_1000_50_activs_sigmoid_sigmoid_sigmoid_sigmoid_sigmoid_sigmoid_lr_0.0001_epcs_154_bs_5_jobs_1_dprs_0.0_0.0_0.0_0.0_0.0_0.0_0'
bn_layer = min([int(term) for term in file_name.split('layers')[1].split('activs')[0].split('_')[1:-1]])


################################################
# Import bottleneck node vectors.
################################################
print(file_name)
print()
print('Importing bottleneck node vectors...')
t0 = time()
BN_vectors = {}

infoFilePath = 'BN_activations/' + file_name + "_BN_activations.ftr"
readFrame = pd.read_feather(infoFilePath, columns=None, use_threads=True)

genes = list(readFrame['genes'])
num_genes = len(genes)

for node in list(readFrame)[1:]:
	vector = list(readFrame[node])
	gene_activs = sorted([[val, genes[j]] for j,val in enumerate(vector)])
	BN_vectors[node] = list(gene_activs)
	
print('Complete! ', datetime.timedelta(seconds=time() - t0))
print()
################################################
# Import gene sets for enrichment.
################################################
print('Importing gene sets for '+str(data_collection)+'...')
t0 = time()
infoFilePath = data_collection + '_features.ftr'
readFrame = pd.read_feather(infoFilePath, columns=None, use_threads=True)

gene_details = {}

for term in list(readFrame):
	gene_details[term] = list([feat for feat in readFrame[term] if not feat == None])

print('Complete! ', datetime.timedelta(seconds=time() - t0))
print()
print('\tNumber of gene sets:', str(len(gene_details)))
print()
############################################################################
# Import precomputed cumulative hypergeometric distribution probabilities.
############################################################################
print('Importing precomputed cumulative hypergeometric distribution probabilities...')
t0 = time()
logPr_dict = {}

try:
	infoFilePath = 'HypergeometricDist.ftr'
	readFrame = pd.read_feather(infoFilePath, columns=None, use_threads=True)
	
	for term in list(readFrame):
		logPr_dict[term] = float(readFrame[term])

except (OSError, FileNotFoundError) as err:
	print(err)
	
print('Complete! ', datetime.timedelta(seconds=time() - t0))
print()
print('\tNumber of precomputed probabilities:', str(len(logPr_dict)))
print()

FDR_thresh = 1e-6
gene_feats = {'Op_neg':0, 'GS_neg':0, 'GO_neg':0, 'eco_neg':0,
             'Op_pos':0, 'GS_pos':0, 'GO_pos':0, 'eco_pos':0}

print('Begin enrichment...')
t0 = time()
for i,node in enumerate([list(BN_vectors)[0]]):
	print('\tNetwork_node:', node)
	
	signi_terms_pos = {}
	signi_terms_neg = {}
	cnt_term_neg = {}
	cnt_term_pos = {}

	node_vec = list(BN_vectors[node])
	node_rev = list(BN_vectors[node][::-1])
	
	for n in range(1, int(num_genes/2)+1):
		n_neg = n
		n_pos = n
		
		# Begin examination of the gene at the forward end of the node vector.
		gene_i = node_vec[n_neg-1][1]
		
		# Retrieve gene sets associated with the current gene in the vector.
		# These include the KEGG pathways, GO processes, operons, and transcription factors.
		terms = set([gene_details[gene_i][4]] +\
				list_format(gene_details[gene_i][7]) +\
				list_format(gene_details[gene_i][8]) +\
				list_format(gene_details[gene_i][9]))
		
		# Count the number of occurrences of each gene set seen thus far.
		for term in terms:
			if not 'None' in term and term in gene_details:
				if term in cnt_term_neg:
					cnt_term_neg[term] += 1
				else:
					cnt_term_neg[term] = 1
		
		# Repeat for the reverse end of the vector
		gene_i = node_rev[n_pos-1][1]
		
		terms = set([gene_details[gene_i][4]] +\
				list_format(gene_details[gene_i][7]) +\
				list_format(gene_details[gene_i][8]) +\
				list_format(gene_details[gene_i][9]))
			
		for term in terms:
			if not 'None' in term and term in gene_details:
				if term in cnt_term_pos:
					cnt_term_pos[term] += 1
				else:
					cnt_term_pos[term] = 1
		
		# Compute the cumulative hypergeometric probability for the gene sets 
		# associated with the current gene on the forward end.
		for term in list(cnt_term_neg):
			k = cnt_term_neg[term]
			K = len(gene_details[term])
			key = str(k) + '_' + str(num_genes) + '_' + str(K) + '_' + str(n_neg)
			if not key in logPr_dict:
				pr = hypergeom.cdf(k-1, num_genes, K, n_neg)
				pr_tot = hypergeom.cdf(int(K*n_neg/num_genes)-1, num_genes, K, n_neg)
				logPr_dict[key] = round(np.log10((1 - pr)/(1 - pr_tot) + 1e-16), 3)

			logPr_neg = logPr_dict[key]
			enrich = int(k/n >= K/num_genes)
			
			if term in signi_terms_neg:
				if -logPr_neg*enrich > signi_terms_neg[term]:
					signi_terms_neg[term] = -logPr_neg*enrich
			else:
				signi_terms_neg[term] = -logPr_neg*enrich
			
		# And for the reverse end.
		for term in list(cnt_term_pos):
			k = cnt_term_pos[term]
			K = len(gene_details[term])
			key = str(k) + '_' + str(num_genes) + '_' + str(K) + '_' + str(n_pos)
			if not key in logPr_dict:
				pr = hypergeom.cdf(k-1, num_genes, K, n_pos)
				pr_tot = hypergeom.cdf(int(K*n_pos/num_genes)-1, num_genes, K, n_pos)
				logPr_dict[key] = round(np.log10((1 - pr)/(1 - pr_tot) + 1e-16), 3)

			logPr_pos = logPr_dict[key]
			enrich = int(k/n >= K/num_genes)
			
			if term in signi_terms_pos:
				if -logPr_pos*enrich > signi_terms_pos[term]:
					signi_terms_pos[term] = -logPr_pos*enrich
			else:
				signi_terms_pos[term] = -logPr_pos*enrich
		
		
	################################################
	# Control the false discovery rate using
	# the Benjamini-Hochberg correction.
	################################################	
	
	sorted_signi = sorted([[-signi_terms_neg[term], term + '_neg'] for term in signi_terms_neg] + [[-signi_terms_pos[term], term + '_pos'] for term in signi_terms_pos])
	BH_len = len(sorted_signi)
	BH_index = BH_len
	p_value = 10**sorted_signi[BH_index - 1][0]
	BH_thresh = FDR_thresh*BH_index/BH_len
	
	while p_value > BH_thresh and BH_index > 1:
		BH_index -= 1
		p_value = 10**sorted_signi[BH_index - 1][0]
		BH_thresh = FDR_thresh*BH_index/BH_len
	
	print('\tNumber of enriched terms:', BH_index)
	signi_terms = list(sorted_signi[:BH_index])
	
	###################################################
	# Save the enriched gene sets as a feather file
	# for each node.
	###################################################
	
	print('\tWriting file ' + file_name + '_FDR_' + str(FDR_thresh) + '_' + node + '.ftr')
	infoFilePath = 'BN_gene_sets/' + file_name + '_FDR_' + str(FDR_thresh) + '_' + node + '.ftr'

	df_list = []
	df_list.append(pd.DataFrame({'neg':["["+str(term[0])+", '"+term[1].replace('_neg', '')+"']" for term in signi_terms if 'neg' in term[1]]}))
	df_list.append(pd.DataFrame({'pos':["["+str(term[0])+", '"+term[1].replace('_pos', '')+"']" for term in signi_terms if 'pos' in term[1]]}))
	dataFrame = pd.concat(df_list, axis=1)
	
	dataFrame.to_feather(infoFilePath)
	print('\tTotal time =', datetime.timedelta(seconds=time() - t0))
	print()
print('Complete! ', datetime.timedelta(seconds=time() - t0))

if store_probs:
	############################################
	# Save computed probabilities.
	############################################
	print('Saving probabilities')
	infoFilePath = file_path + 'HypergeometricDist.ftr'
	dataFrame = pd.DataFrame(data=logPr_dict, index=[0])
	dataFrame.to_feather(infoFilePath)



