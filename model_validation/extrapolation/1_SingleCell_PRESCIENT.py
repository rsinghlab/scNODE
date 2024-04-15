'''
Description:
    Run PRESCIENT on extrapolating multiple timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np
from datetime import datetime

from model_validation.extrapolation.utils import loadSCData, tunedPRESCIENTPars
from baseline.prescient_model.process_data import main as prepare_data
from baseline.prescient_model.running import prescientTrain, prescientSimulate

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")


# ======================================================
# Load data and pre-processing
print("=" * 70)
data_name = "wot"
# The data are available at https://doi.org/10.6084/m9.figshare.25607679s
if data_name == "zebrafish":
    data_dir = "./extrapolate_data/zebrafish_embryonic/"
elif data_name == "drosophila":
    data_dir = "./extrapolate_data/drosophila_embryonic/"
elif data_name == "wot":
    data_dir = "./extrapolate_data/Schiebinger2019/"
else:
    raise ValueError("Unknown data name {}!".format(data_name))

for forecast_num in [1, 2, 3, 4, 5]:
    # forecast_num = 1
    split_type = "forecasting_{}".format(forecast_num)
    print("[ {} ]".format(data_name).center(60))
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type, data_dir=data_dir)
    train_tps = list(range(n_tps - forecast_num))
    test_tps = list(range(n_tps - forecast_num, n_tps))
    data = ann_data.X
    all_tps = list(np.arange(n_tps))

    processed_data = ann_data[
        [t for t in range(ann_data.shape[0]) if (ann_data.obs["tp"].values[t] - 1.0) in train_tps]
    ]
    cell_tps = (processed_data.obs["tp"].values - 1.0).astype(int)
    cell_types = np.repeat("NAN", processed_data.shape[0])
    genes = processed_data.var.index.values

    # Parameter settings
    k_dim, layers, sd, tau, clip = tunedPRESCIENTPars(data_name, forecast_num)

    # PCA and construct data dict
    data_dict, scaler, pca, um = prepare_data(
        processed_data.X, cell_tps, cell_types, genes,
        num_pcs=k_dim, num_neighbors_umap=10
    )
    data_dict["y"] = list(range(train_tps[-1] + 1))
    data_dict["x"] = [data_dict["x"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
    data_dict["xp"] = [data_dict["xp"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
    data_dict["w"] = [data_dict["w"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]

    # ======================================================
    # Model training
    train_epochs = 2000
    train_lr = 1e-3
    final_model, best_state_dict, config, loss_list = prescientTrain(
        data_dict, data_name=data_name, out_dir="", train_t=train_tps[1:], timestamp=timestamp,
        k_dim=k_dim, layers=layers, train_epochs=train_epochs, train_lr=train_lr,
        train_sd=sd, train_tau=tau, train_clip=clip
    )

    # Simulation
    n_sim_cells = 2000
    sim_data = prescientSimulate(
        data_dict,
        data_name=data_name,
        best_model_state=best_state_dict,
        num_cells=n_sim_cells,
        num_steps=(n_tps - 1) * 10,  # (#tps - 1) * 10, as dt=0.1
        config=config
    )
    sim_tp_latent = [sim_data[int(t * 10)] for t in range(len(all_tps))]  # dt=0.1 in all cases
    sim_tp_recon = [scaler.inverse_transform(pca.inverse_transform(each)) for each in sim_tp_latent]

    # ======================================================
    # Save results
    res_filename = "./res/extrapolation/{}-{}-PRESCIENT-res.npy".format(data_name, split_type)
    print("Saving to {}".format(res_filename))
    res_dict = {
        "true": [ann_data.X[(ann_data.obs["tp"].values - 1.0) == t] for t in range(len(all_tps))],
        "pred": sim_tp_recon,
        "latent_seq": sim_tp_latent,
        "tps": {"all": all_tps, "train": train_tps, "test": test_tps},
    }
    res_dict["true_pca"] = [pca.transform(scaler.transform(each)) for each in res_dict["true"]]
    np.save(res_filename, res_dict, allow_pickle=True)

    # save model and config
    model_dir = "./res/extrapolation/{}-{}-PRESCIENT-state_dict.pt".format(data_name, split_type)
    config_dir = "./res/extrapolation/{}-{}-PRESCIENT-config.pt".format(data_name, split_type)
    torch.save(best_state_dict['model_state_dict'], model_dir)
    torch.save(config.__dict__, config_dir)

