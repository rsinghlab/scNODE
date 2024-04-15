# Description: Extract highly variable genes of zebrafish embryonic (ZB) scRNA-seq data.
# Author: Jiaqi Zhang <jiaqi_zhang2@brown.edu>
# Reference:
#   https://www.science.org/doi/10.1126/science.aar3131
#   https://github.com/farrellja/URD
#   https://singlecell.broadinstitute.org/single_cell/study/SCP162/
library(URD)
library(Seurat)
library(stringr)

# -----
# Load data
print("Loading zebrafish data...")
data_obj <- readRDS("./data/single_cell/experimental/zebrafish_embryonic/raw/URD_Zebrafish_Object.rds")
data_count <- data_obj@count.data # gene x cell
meta_df <- data_obj@meta
print(sprintf("Data shape (gene x cell): %d x %d", dim(data_count)[1], dim(data_count)[2]))
# -----
# Split data by traing and testing sets
split_type <- "remove_recovery" # two_forecasting, three_interpolation, remove_recovery
cell_tp <- meta_df$stage.nice
unique_tp <- unique(cell_tp)
if (split_type == "three_interpolation"){ # easy
  train_tps <- c(1, 2, 3, 4, 6, 8, 10, 11, 12)
  test_tps <- c(5, 7, 9)
} else if (split_type == "two_forecasting"){ # medium
  train_tps <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  test_tps <- c(11, 12)
} else if (split_type == "remove_recovery"){ # hard
  train_tps <- c(1, 2, 4, 6, 8, 10)
  test_tps <- c(3, 5, 7, 9, 11, 12)
} else {
  stop(sprintf("Unknown split type %s!", split_type))
}
print(sprintf("Num tps: %d", length(unique_tp)))
print("Train tps:")
print(train_tps)
print("Test tps:")
print(test_tps)
train_data <- data_count[, which(cell_tp %in% unique_tp[train_tps])]
# test_data <- data_count[, which(cell_tp %in% unique_tp[test_tps])]
print(sprintf("Train data shape (gene x cell): %d x %d", dim(train_data)[1], dim(train_data)[2]))
# print(sprintf("Test data  shape (gene x cell): %d x %d", dim(test_data)[1], dim(test_data)[2]))
# -----
# Select highly variables based on training data
train_obj <- CreateSeuratObject(counts=train_data)
train_obj <- FindVariableFeatures(NormalizeData(train_obj), selection.method = "vst", nfeatures = 2000) # use log-normalized data
hvgs <- VariableFeatures(train_obj)
hvg_diff <- hvgs[!(hvgs %in% row.names(data_count))]
if (length(hvg_diff) > 0){ # solve the problem that replacing "_" with "-"
  print("Tune hvg list...")
  hvgs <- c(hvgs[hvgs %in% row.names(data_count)], str_replace(hvg_diff, "-", "_"))
}
data_count_hvg <- data_count[hvgs, ]
print(sprintf("HVG data shape (gene x cell): %d x %d", dim(data_count_hvg)[1], dim(data_count_hvg)[2]))

write.csv(as.matrix(t(data_count_hvg)), sprintf("./data/single_cell/experimental/zebrafish_embryonic/new_processed/%s-count_data-hvg.csv", split_type))
write.csv(hvgs, sprintf("./data/single_cell/experimental/zebrafish_embryonic/new_processed/%s-var_genes_list.csv", split_type))
write.csv(meta_df, "./data/single_cell/experimental/zebrafish_embryonic/new_processed/meta_data.csv")
