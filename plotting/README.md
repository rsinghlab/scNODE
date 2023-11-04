# plotting

Codes for plotting figures in the paper. Compare and visualize model predictions. `latent_ODE_OT_pretrain` refers to scNODE.

All model predictions on six datasets are available at [here](https://figshare.com/articles/dataset/Model_predictions_/24493732). 

Other experimental results, including evaluation metrics, ablation study, and downstream analysis can be downloaded from [here](https://figshare.com/articles/dataset/Other_experiment_results_/24493459).

In the paper:
- Fig. 2 is generated from `plotMetricBar` in [./Compare_SingleCell_Predictions.py](./Compare_SingleCell_Predictions.py)
- Fig. 3 is generated from `compareUMAPTestTime` in [./Compare_SingleCell_Predictions.py](./Compare_SingleCell_Predictions.py)
- Fig. 4 is generated from [../downstream_analysis/1_Perturbation_Analysis.py](../downstream_analysis/1_Perturbation_Analysis.py)
- Fig. 5 is generated from [../downstream_analysis/2_Recover_Trajectory.py](../downstream_analysis/2_Recover_Trajectory.py)
