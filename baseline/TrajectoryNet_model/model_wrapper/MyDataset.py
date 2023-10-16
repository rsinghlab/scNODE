'''
Description:
    Prepare data for running TrajectoryNet in our benchmark.

Reference:
    [1] https://github.com/KrishnaswamyLab/TrajectoryNet/blob/master/TrajectoryNet/dataset.py
'''
import numpy as np
import scanpy
import pandas as pd
import natsort
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from baseline.TrajectoryNet_model.TrajectoryNet.dataset import SCData
from benchmark.BenchmarkUtils import preprocess

# ==================================================

def loadPancreaticData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadEmbryoidData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


class BenchmarkData(SCData):
    def __init__(
        self, dir_path, data_name, split_type, embedding_name="pca", n_pcs=50, use_velocity=False
    ):
        super().__init__()
        self.data_name = data_name
        self.dir_path = dir_path
        self.split_type = split_type
        self.embedding_name = embedding_name
        self.n_pcs = n_pcs
        self.pca_model = PCA(n_components=n_pcs)
        self.scalar_model = StandardScaler()
        self.use_velocity = use_velocity
        self.load()


    def load(self):
        if self.data_name == "pancreatic":
            ann_data = loadPancreaticData(self.dir_path, self.split_type)
        elif self.data_name == "embryoid":
            ann_data = loadEmbryoidData(self.dir_path, self.split_type)
        else:
            raise ValueError("Unknown data name {}!".format(self.data_name))
        ann_data.X = ann_data.X.astype(float)
        ann_data = preprocess(ann_data.copy())
        self.labels = ann_data.obs["tp"].values - 1.0
        self.true_data = ann_data
        expr_mat = ann_data.X
        # PCA
        embedding = self.pca_model.fit_transform(expr_mat)
        # Normalization
        self.data = self.scalar_model.fit_transform(embedding)
        self.ncells = embedding.shape[0]
        assert self.labels.shape[0] == self.ncells



    def has_velocity(self):
        return False

    def known_base_density(self):
        return False

    def get_data(self):
        return self.data

    def get_times(self):
        return self.labels

    def get_unique_times(self):
        return np.unique(self.labels)

    def get_velocity(self):
        return self.velocity

    def get_shape(self):
        return [self.data.shape[1]]

    def get_ncells(self):
        return self.ncells

    def leaveout_timepoint(self, tp):
        """Takes a timepoint label to leaveout
        Alters data stored in object to leave out all data associated
        with that timepoint.
        """
        if tp < 0:
            raise RuntimeError("Cannot leaveout negative timepoint %d." % tp)
        mask = self.labels != tp
        print("Leaving out %d samples from sample %d" % (np.sum(~mask), tp))
        self.labels = self.labels[mask]
        self.data = self.data[mask]
        self.velocity = self.velocity[mask]
        self.ncells = np.sum(mask)

    def sample_index(self, n, label_subset):
        arr = np.arange(self.ncells)[self.labels == label_subset]
        return np.random.choice(arr, size=n)
