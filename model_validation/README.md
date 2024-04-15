# model_validation

Validate design of our scNODE model.

The preprocessed data can be downloaded from [here](https://doi.org/10.6084/m9.figshare.25601610.v1).
You can put preprocessed data in the `data` directory, otherwise, you should specify the data file path when calling the `benchmark.BencahmarkUtils.loadSCData` function.


## scNODE improvements vs. data distribution shifts

We validate that scNODE is more robust to distribution shift when testing timepoints have substantially different distributions from training data.  
Details may refer to Sec. 3.2 in the paper.
Codes are available in [./distribution_shift](./distribution_shift).


## Evaluate model predictions when extrapolating more timepoints

We evaluate model predictions in extrapolating multiple timepoints. Details may refer to Supplementary Sec. 6.1 in the paper.
Codes are available in [./extrapolation](./extrapolation).


## Ablation study of scNODE pretraining

We compare the performance of scNODE predictions when
excluding the pre-training step and using different numbers of training timepoints in pre-training. 
Details may refer to Supplementary Sec. 6.3 in the paper.
Codes are available in [./pretraining](./pretraining).


## scNODE sensitiveness against latent space size

We test scNODE when varying the latent dimension from {25, 50, 75, · · · , 200}. Details may refer to Supplementary Sec. 6.3.
Codes are available in [./latent_size](./latent_size).


## scNODE sensitiveness against regularization coefficient ($\beta$)

scNODE uses a dynamic regularizer with hyperparameter $\beta$ to update the VAE space dynamically such that it captures
both cellular variations and the developmental dynamics of the scRNA-seq data. We test scNODE predictions when
using different $\beta \in$ {0.25, 0.5, · · · , 10.0}. Details may refer to Supplementary Sec. 6.3.
Codes are available in [./tune_reg_coeff](./tune_reg_coeff).
