'''
Description:
    Figure plotting for the learned latent space.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from plotting import *
from plotting import _removeTopRightBorders, _removeAllBorders, linearSegmentCMap

# ======================================================

base_dir = "../res/adjustment/latent/"

# ======================================================
bg_s_size = 5
s_size = 1

def _plotScatter(ax, db, milisi, umap_embed, test_tps, color_list, title="Pre-Trained"):
    # ax.set_title("{} \n DB={:.2f}, miLISI={:.2f}".format(title, db, milisi),fontsize=12)
    ax.set_title("{}".format(title, db, milisi),fontsize=18)
    comcat_umap_emb = np.concatenate(umap_embed, axis=0)
    ax.scatter(
        comcat_umap_emb[:, 0], comcat_umap_emb[:, 1],
        color=gray_color, s=bg_s_size, alpha=0.5
    )
    for t in test_tps:
        c = color_list[test_tps.index(t)]
        ax.scatter(
            umap_embed[t][:, 0], umap_embed[t][:, 1],
            color=c, s=s_size, alpha=1.0
        )
        ax.scatter(
            [], [],
            label=int(t), color=c, s=50, alpha=1.0
        )



def mainCompareLatentPCAReduced(data_name, split_type, trial_id):
    res_filename = "{}/{}-{}-ALL-PCA_emb_and_score-trial{}.npy".format(base_dir, data_name, split_type, trial_id)
    res_dict = np.load(res_filename, allow_pickle=True).item()
    print("Loading for scNODE...")
    scNODE_pretrain_emb, scNODE_final_emb = res_dict["scNODE_pretrain_emb"], res_dict["scNODE_final_emb"]
    scNODE_pretrain_db, scNODE_pretrain_milisi, scNODE_final_db, scNODE_final_milisi = res_dict["scNODE_pretrain_db"], res_dict["scNODE_final_milisi"], res_dict["scNODE_final_db"], res_dict["scNODE_final_milisi"]
    # -----
    print("Loading for MIOFlow...")
    MIOFlow_pretrain_emb, MIOFlow_final_emb = res_dict["MIOFlow_pretrain_emb"], res_dict["MIOFlow_final_emb"]
    MIOFlow_pretrain_db, MIOFlow_pretrain_milisi, MIOFlow_final_db, MIOFlow_final_milisi = res_dict["MIOFlow_pretrain_db"], res_dict["MIOFlow_pretrain_milisi"], res_dict["MIOFlow_final_db"], res_dict["MIOFlow_final_milisi"]
    # -----
    print("Loading for PRESCIENT...")
    PRESCIENT_pretrain_emb, PRESCIENT_final_emb = res_dict["PRESCIENT_pretrain_emb"], res_dict["PRESCIENT_final_emb"]
    PRESCIENT_pretrain_db, PRESCIENT_pretrain_milisi, PRESCIENT_final_db, PRESCIENT_final_milisi = res_dict["PRESCIENT_pretrain_db"], res_dict["PRESCIENT_pretrain_milisi"], res_dict["PRESCIENT_final_db"], res_dict["PRESCIENT_final_milisi"]
    # -----
    # color_list = linearSegmentCMap(n_tps, "viridis")
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, 3, figsize=(7, 2))
    _plotScatter(ax_list[0], scNODE_final_db, scNODE_final_milisi, scNODE_final_emb, test_tps, color_list, title="scNODE")
    _plotScatter(ax_list[1], MIOFlow_final_db, MIOFlow_final_milisi, MIOFlow_final_emb, test_tps, color_list, title="MIOFlow")
    _plotScatter(ax_list[2], PRESCIENT_final_db, PRESCIENT_final_milisi, PRESCIENT_final_emb, test_tps, color_list, title="PRESCIENT")
    for ax in ax_list:
        ax.set_xticks([])
        ax.set_yticks([])
        _removeAllBorders(ax)
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Test TPs", title_fontsize=15, fontsize=14)
    plt.tight_layout()
    plt.show()


# ======================================================

if __name__ == '__main__':
    # Configuration
    print("=" * 70)
    data_name = "wot"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    forecast_num = 3
    split_type = "forecasting_{}".format(forecast_num)
    # split_type = "remove_recovery"
    print("Split type: {}".format(split_type))
    trial_id = 0
    if data_name == "zebrafish":
        n_tps = 12
    if data_name == "drosophila":
        n_tps = 11
    if data_name == "wot":
        n_tps = 19
    train_tps = list(range(n_tps - forecast_num))
    test_tps = list(range(n_tps - forecast_num, n_tps))
    mainCompareLatentPCAReduced(data_name, split_type, trial_id)