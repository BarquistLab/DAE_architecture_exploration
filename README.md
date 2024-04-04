# DAE architecture exploration

Pipeline for the exploration of denoising autoencoder (DAE) architectures performed in the paper 'Network depth affects inference of gene sets from bacterial transcriptomes using denoising autoencoders.' If you use this analysis, please cite our preprint.

[doi.org/10.1101/2023.05.30.542622](https://doi.org/10.1101/2023.05.30.542622)

## Training DAEs on a bacterial gene expression compendium

To train a DAE, simply run the following from the commandline:

> python DAE_train.py 0

The index (0) informs the python script which random training / validation / test split of the compendium the DAE will be trained on. There are 10 different splits found in the Data/ folder. The remaing parameters relevant for our exploration, including the training parameters, are specified in the DAE_params.py script. 

Note that upon completion, both the final network, and the training loss trajectories are saved to the Networks/ and Training_trajs/ folders, respectively.

## Plotting training trajectories to determine the optimal number of epochs

This script can be run even if the above step hasn't been taken; there are already 10 trajectory files provided in the Training\_trajs/ folder. The network architecture and training parameters corresponding to the trajectories to plot are specified in the DAE_params.py script. To plot the already provided results, the variable epcs must be set to 300 in this script, then the following can be run:

> python DAE\_plot\_trajectories.py

This code not only produces a figure (figure S4 in the supplemental material) in the Figures/ folder showing the training trajectories, but it also calculates the optimal number of training epochs based on the minimum of the average validation loss, and prints this to the terminal.
Alternatively, additional trajectory files can be produced if desired by running DAE_train.py as specified in the last section. 

If the optimal number of epochs for early stopping is determined to be much less than the chosen number of epochs, you may wish to train a new ensemble of DAEs for a shorter time before proceeding to the network analysis steps below. For instance, the provided trajectories will result in an optimal early stopping point of 166 epochs. Therefore it is recommended to train new networks with the epcs variable set to 166 in the DAE_params.py script.

## Activating bottleneck nodes and determining associated predicted gene expression vectors

Next, we'd like to determine what each of the nodes at the bottleneck layer of the trained networks represents. To do this, we activate each node individually, and propogate this activation through the decoder of each network. The resulting changes in predicted gene expression are then saved to vectors for further analysis.

To perform this step, simply run

> python DAE\_activate\_nets.py

10 example networks of the 2000-1000-(50) architecture are already provided in the Networks/ folder. These networks were run to 154 epochs, and so this parameter (epcs) must be changed in the DAE_params.py script before this step can be run. The resulting vectors are saved in the BN\_activations/ folder.

Now that we have these bottleneck node vectors, we'd like to perform enrichment analysis to determine their biological significance. For this, we compute the probabilities of each enriched gene set based on the hypergeometric distribution, and then correct for multiple hypotheis testing using the Benjamini-Hochberg method. 

A collection of gene sets for enrichment are already provided in the E\_coli\_PRECISE2\_features.ftr file. We have also provided these same gene sets in the file Known\_gene\_sets.csv so that users can easily explore their contents.

Note that which network ensemble to examine is specified by the file_name variable, and is currently set to look for an ensemble of 10 2000-1000-(50) networks trained to 154 epochs.

The rate-limiting step in this computation is the calculation of the cumulative hypergeometric distribution probability. To alleviate this computational burden, we have created the option to save any computed probabilities. To utilize this option, set the variable store_probs to True in DAE\_analyze\_compression.py. Additionally, further terms are stored in memory during the enrichment calculation, so the first node takes the longest computation time, and each subsequent node is faster than the last.

To perform the enrichment analysis, run the following:

> python DAE\_analyze\_compression.py

Once this analysis is complete, the results will be printed to the terminal, and additionally saved in the BN\_gene\_sets/ folder. 

