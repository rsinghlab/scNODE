# scNODE: Generative Model for Temporal Single Cell Transcriptomic Data Prediction

scNODE is a generative model that simulates and predicts realistic *in silico* single-cell gene expressions at any timepoint. scNODE integrates the idea of variational autoencoder (VAE) and neural ordinary differential equation (ODE) to model cell developmental landscapes on the non-linear manifold. scNODE is scalable to large-scale datasets.

![scNODE model overview](https://github.com/rsinghlab/scNODE/blob/main/model_illustration.jpg?raw=true)

**If you have questions or find any problems with our codes, feel free to submit issues or send emails to jiaqi_zhang2@brown.edu or other corresponding authors.**



## Requirements

Our codes have been tested in Python 3.7. Required packages are listed in [./installation](./installation).

## Data

- Raw and preprocessed data of six scRNA-seq datasets can be downloaded from [here](https://figshare.com/articles/dataset/Raw_and_processed_data_of_six_scRNA-seq_datasets_/24493369).
- All model predictions on six datasets are available at [here](https://figshare.com/articles/dataset/Model_predictions_/24493732).
- Other experimental results, including evaluation metrics, ablation study, and downstream analysis can be downloaded from [here](https://figshare.com/articles/dataset/Other_experiment_results_/24493459).



## Models

scNODE is implemented in [./model/dynamic_model.py](./model/dynamic_model.py). We also provide codes of baseline models in [./baseline](./baseline).


## Example Usage

An example of using scNODE is shown in [./benchmark/1_SingleCell_scNODE.py](./benchmark/1_SingleCell_scNODE.py).


## Repository Structure

- [data](./data): Scripts for data preprocessing. Some scripts are implemented in R and need installation of [Seurat](https://satijalab.org/seurat/).
- [model](./model): Implementation of scNODE model.
- [optim](./optim): Loss computations and evaluation metrics.
- [baseline](./baseline): Implementation of baseline models.
- [benchmark](./benchmark): Run each model on six scRNA-seq datasets and make predictions. Test scNODE performance with different settings.
- [downstream_analysis](./downstream_analysis): Use scNODE for perturbation analysis and help recapitulate smooth trajectories.
- [plotting](./plotting): Prediction visualization. Compare model predictions.



## Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [Github issue](https://github.com/rsinghlab/scNODE/issues)
