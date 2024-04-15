# benchmark

Compare our scNODE model with baselines on three scRNA-seq datasets.

The preprocessed data can be downloaded from [here](https://doi.org/10.6084/m9.figshare.25601610.v1).
You can put preprocessed data in the `data` directory, otherwise, you should specify the data file path in [./BenchmarkUtils.py](./BenchmarkUtils.py).

## Model running

- [1_SingleCell_scNODE.py](./1_SingleCell_scNODE.py): Run scNODE model.
- [2_SingleCell_PRESCIENT.py](./2_SingleCell_PRESCIENT.py): Run PRESCIENT model.
- [3_SingleCell_MIOFlow.py](./3_SingleCell_MIOFlow.py): Run MIOFlow model.

Comparison of predictions are provided in [../plotting/Compare_SingleCell_Predictions.py](../plotting/Compare_SingleCell_Predictions.py).
