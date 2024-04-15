# Description: Extract highly variable genes of drosophila (DR) scRNA-seq data.
# Author: Jiaqi Zhang <jiaqi_zhang2@brown.edu>
# Reference:
#   https://www.science.org/doi/10.1126/science.abn5800
#   https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/
library(Seurat)
library(stringr)

subsampleCell <- function(num_cells, sample_ratio) {
  num_sample <- floor(num_cells * sample_ratio)
  sample_idx <- sample(x = 1:num_cells, size = num_sample, replace = FALSE)
  return(sample_idx)
}

# # Split count data for every time point
# # Reference: https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/scripts/RNA/data_processing/
# data <- readRDS("main.Rds")
# times <- unique(data@meta.data$time)
# sapply(times, function(a_time) {
#     print(paste0('starting on time window: ', a_time))
# 	subdata <- subset(data, subset= time == a_time) # check that this works
#     saveRDS(subdata, sprintf("./data/single_cell/experimental/drosophila_embryonic/raw/%s_finished_processing_count.Rds", a_time))
#     print('done')
# })

# -----
# Load data and subsampling
print("Loading drosophila data...")
window_list <- c("00_02", "01_03", "02_04", "03_07", "04_08", "06_10", "08_12", "10_14", "12_16", "14_18", "16_20")
if (file.exists("./data/single_cell/experimental/drosophila_embryonic/processed/subsampled_data.rds")) {
  print("Subsampled data already exist. Loading data...")
  all_data_obj <- readRDS("./data/single_cell/experimental/drosophila_embryonic/processed/subsampled_data.rds") # gene x cell
  print(sprintf("Data shape (gene x cell): %d x %d", dim(all_data_obj)[1], dim(all_data_obj)[2]))
} else {
  sampling_rate <- 0.05
  set.seed(111)
  first_data_obj <- readRDS(sprintf("./data/single_cell/experimental/drosophila_embryonic/raw/hrs_%s_finished_processing_count.Rds", window_list[1])) # gene x cell
  subsample_idx <- subsampleCell(dim(first_data_obj)[2], sampling_rate)
  first_data_obj <- first_data_obj[, subsample_idx]
  VariableFeatures(first_data_obj) <- c()
  t_data_list <- c()
  for (t in 2:length(window_list)) {
    t_data_obj <- readRDS(sprintf("./data/single_cell/experimental/drosophila_embryonic/raw/hrs_%s_finished_processing_count.Rds", window_list[t])) # gene x cell
    subsample_idx <- subsampleCell(dim(t_data_obj)[2], sampling_rate)
    t_data_obj <- t_data_obj[, subsample_idx]
    VariableFeatures(t_data_obj) <- c()
    t_data_list <- c(t_data_list, t_data_obj)
  }
  all_data_obj <- merge(first_data_obj, y = t_data_list)
  print(sprintf("Data shape (gene x cell): %d x %d", dim(all_data_obj)[1], dim(all_data_obj)[2]))
  saveRDS(all_data_obj, "./data/single_cell/experimental/drosophila_embryonic/processed/subsampled_data.rds") # gene x cell
  write.csv(all_data_obj@meta.data, "./data/single_cell/experimental/drosophila_embryonic/processed/subsample_meta_data.csv")
}
# -----
# Split data by traing and testing sets
split_type <- "remove_recovery" # three_forecasting, three_interpolation, remove_recovery
cell_tp <- all_data_obj@meta.data$time
unique_tp <- unique(cell_tp)
if (split_type == "three_forecasting"){ # medium
  train_tps <- c(1, 2, 3, 4, 5, 6, 7, 8)
  test_tps <- c(9, 10, 11)
} else if (split_type == "three_interpolation"){ # easy
  train_tps <- c(1, 2, 3, 4, 6, 8, 10, 11)
  test_tps <- c(5, 7, 9)
} else if (split_type == "remove_recovery"){ # hard
  train_tps <- c(1, 2, 4, 6, 8)
  test_tps <- c(3, 5, 7, 9, 10, 11)
} else {
  stop(sprintf("Unknown split type %s!", split_type))
}
print(sprintf("Num tps: %d", length(unique_tp)))
print("Train tps:")
print(train_tps)
print("Test tps:")
print(test_tps)
train_obj <- all_data_obj[, which(cell_tp %in% unique_tp[train_tps])]
print(sprintf("Train data shape (gene x cell): %d x %d", dim(train_obj)[1], dim(train_obj)[2]))
# -----
# Select highly variables based on training data
train_obj <- FindVariableFeatures(NormalizeData(train_obj), selection.method = "vst", nfeatures = 2000)
hvgs <- VariableFeatures(train_obj)
data_count_hvg <- GetAssayData(all_data_obj[["RNA"]], slot="counts")[hvgs, ]
print(sprintf("HVG data shape (gene x cell): %d x %d", dim(data_count_hvg)[1], dim(data_count_hvg)[2]))
# -----
# Save data
write.csv(t(as.matrix(data_count_hvg)), sprintf("./data/single_cell/experimental/drosophila_embryonic/processed/%s-count_data-hvg.csv", split_type))
write.csv(hvgs, sprintf("./data/single_cell/experimental/drosophila_embryonic/processed/%s-var_genes_list.csv", split_type))
