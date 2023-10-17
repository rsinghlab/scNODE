# benchmark

Compare our scNODE model with baselines on six scRNA-seq datasets.

## Model running

- [1_SingleCell_scNODE.py](./1_SingleCell_scNODE.py): Run scNODE model.
- [2_SingleCell_PRESCIENT.py](./2_SingleCell_PRESCIENT.py): Run PRESCIENT model.
- [3_SingleCell_WOT.py](./3_SingleCell_WOT.py): Run WOT model.
- [4_SingleCell_TrajectoryNet.py](./4_SingleCell_TrajectoryNet.py): Run TrajectoryNet model.
- [5_SingleCell_Dummy.py](./5_SingleCell_Dummy.py): Run Dummy model.

Comparison of predictions are  provided in [../plotting/Compare_SingleCell_Predictions.py](../plotting/Compare_SingleCell_Predictions.py).

## scNODE Sensitiveness against Hyperparameter Settings

- [6_Performance_vs_LatentSize.py](./6_Performance_vs_LatentSize.py): Test scNODE performance with different latent size.
- [6_Performance_vs_LatentCoeff.py](./6_Performance_vs_LatentCoeff.py): Test scNODE performance with different regularization coefficient beta.
- [6_Performance_vs_Train_TPs.py](./6_Performance_vs_Train_TPs.py): Test scNODE performance with different number of training timepoints.

## Time Cost

We record the time cost of scNODE and PRESCIENT in [7_TimeCost_scNODE.py](./7_TimeCost_scNODE.py) and 
[7_TimeCost_PRESCIENT.py](./7_TimeCost_PRESCIENT.py) correspondingly. 
Visualization may refer to [../plotting/Comapre_TimeCost.py](../plotting/Comapre_TimeCost.py).