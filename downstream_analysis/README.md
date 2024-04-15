# downstream analysis

This directory contains experiments for two downstream analysis tasks. 


## scNODE predictions help recover cell trajectories

We carry out trajectory analysis on both real and predicted single-cell gene expression, to determine if scNODE can aid with temporal downstream analysis.
We compare scNODE with PRESCIENT and MIOFlow.
Details may refer to Sec. 3.3 of our paper.
Need to install [networkx](https://pypi.org/project/networkx/) and [netrd](https://github.com/netsiphd/netrd) for this task.
 



## scNODE assists with perturbation analysis

We conduct *in silico* perturbation analysis with the latent space learned by scNODE on ZB dataset.
Details may refer to Sec. 3.4 of our paper.
