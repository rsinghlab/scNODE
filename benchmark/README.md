# benchmark

Compare our scNODE model with baselines on six scRNA-seq datasets.

## Model running

- [1_SingleCell_scNODE.py](./1_SingleCell_scNODE.py): Run scNODE model.
- [2_SingleCell_PRESCIENT.py](./2_SingleCell_PRESCIENT.py): Run PRESCIENT model.
- [3_SingleCell_WOT.py](./3_SingleCell_WOT.py): Run WOT model.
- [4_SingleCell_TrajectoryNet.py](./4_SingleCell_TrajectoryNet.py): Run TrajectoryNet model.
- [5_SingleCell_Dummy.py](./5_SingleCell_Dummy.py): Run Dummy model.

## scNODE Sensitiveness against Hyperparameter Settings

- [6_Performance_vs_LatentSize.py](./6_Performance_vs_LatentSize.py): Test scNODE performance with different latent size.
- [6_Performance_vs_LatentCoeff.py](./6_Performance_vs_LatentCoeff.py): Test scNODE performance with different regularization coefficient beta.
- [6_Performance_vs_Train_TPs.py](./6_Performance_vs_Train_TPs.py): Test scNODE performance with different number of training timepoints.

## Time Cost