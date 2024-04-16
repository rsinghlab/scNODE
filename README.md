# scNODE: Generative Model for Temporal Single Cell Transcriptomic Data Prediction

scNODE is a generative model that simulates and predicts realistic *in silico* single-cell gene expressions at any timepoint. scNODE integrates the idea of variational autoencoder (VAE) and neural ordinary differential equation (ODE) to model cell developmental landscapes on the non-linear manifold. scNODE is scalable to large-scale datasets.
[(bioRxiv preprint)](https://www.biorxiv.org/content/10.1101/2023.11.22.568346v2)

![scNODE model overview](https://github.com/rsinghlab/scNODE/blob/main/model_illustration.jpg?raw=true)

**If you have questions or find any problems with our codes, feel free to submit issues or send emails to jiaqi_zhang2@brown.edu or other corresponding authors.**

*(04/15/2024 updates) We have revised the codes to align with the updated paper.* 

## Requirements

Our codes have been tested in Python 3.7. Required packages are listed in [./installation](./installation).

## Data

- Raw and preprocessed data of three scRNA-seq datasets can be downloaded from [here](https://doi.org/10.6084/m9.figshare.25601610.v1).
- All model predictions on three datasets are available at [here](https://doi.org/10.6084/m9.figshare.25602000).
- Experiment results for downstream analysis are available at [here](https://doi.org/10.6084/m9.figshare.25602672).
- Other experimental results, including evaluation metrics, ablation study, and investigation of hyperparameter settings can be downloaded from [here](https://doi.org/10.6084/m9.figshare.25607973).



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
- [downstream_analysis](./downstream_analysis): Use scNODE for perturbation analysis and help recover smooth trajectories.
- [model_validation](./model_validation): Ablation study and investigation of hyperparameter settings.
- [plotting](./plotting): Prediction visualization. Compare model predictions.
- [paper_figs](./paper_figs): Figure plotting for the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.11.22.568346v2).



## Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [Github issue](https://github.com/rsinghlab/scNODE/issues)

