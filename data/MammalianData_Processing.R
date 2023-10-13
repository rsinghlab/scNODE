# Description: Extract highly variable genes of mammalian cerebral cortex scRNA-seq data.
# Author: Jiaqi Zhang <jiaqi_zhang2@brown.edu>
# Reference:
#   https://singlecell.broadinstitute.org/single_cell/study/SCP1290/
#   https://pubmed.ncbi.nlm.nih.gov/34163074/
#   https://github.com/ehsanhabibi/MolecularLogicMouseNeoCortex

library(Seurat)

expr_mat <- ReadMtx(
  mtx= "./data/single_cell/experimental/mammalian_cerebral_cortex/raw/gene_sorted-matrix.mtx.gz",
  features = "./data/single_cell/experimental/mammalian_cerebral_cortex/raw/genes.tsv",
  cells = "./data/single_cell/experimental/mammalian_cerebral_cortex/raw/barcodes.tsv"
)
print("Finished loading expression matrix")
print("Finished creating seurat object")
meta_df <- read.csv("./data/single_cell/experimental/mammalian_cerebral_cortex/raw/metaData_scDevSC.txt", sep="\t")
meta_df <-  meta_df[2:dim(meta_df)[1], ]

print(sprintf("Data shape (gene x cell): %d x %d", dim(expr_mat)[1], dim(expr_mat)[2]))
# -----
# Split data by traing and testing sets
split_type <- "first_five" # interpolation, forecasting, three_forecasting, three_interpolation, all
cell_tp <- meta_df$orig_ident
unique_tp <- unique(cell_tp)
if (split_type == "interpolation"){
  train_tps <- c(1, 2, 3, 4, 5, 6, 9, 10)
  test_tps <- c(7, 8, 11, 12, 13)
} else if (split_type == "forecasting"){
  train_tps <- c(1, 2, 3, 4, 5, 6, 7, 8)
  test_tps <- c(9, 10, 11, 12, 13)
} else if (split_type == "three_forecasting"){
  train_tps <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  test_tps <- c(11, 12, 13)
} else if (split_type == "three_interpolation"){
  train_tps <- c(1, 2, 3, 4, 6, 8, 10, 11, 12, 13)
  test_tps <- c(5, 7, 9)
} else if (split_type == "all"){
  train_tps <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
  test_tps <- c()
} else if (split_type == "first_five"){
  train_tps <- c(1, 2, 3, 4, 5)
  test_tps <- c(6, 7, 8, 9, 10, 11, 12, 13)
} else {
  stop(sprintf("Unknown split type %s!", split_type))
}
print(sprintf("Num tps: %d", length(unique_tp)))
print("Train tps:")
print(train_tps)
print("Test tps:")
print(test_tps)
train_data <- expr_mat[, which(cell_tp %in% unique_tp[train_tps])]
print(sprintf("Train data shape (gene x cell): %d x %d", dim(train_data)[1], dim(train_data)[2]))
# -----
# Select highly variables based on training data
train_obj <- CreateSeuratObject(counts=train_data)
train_obj <- FindVariableFeatures(train_obj, selection.method = "vst", nfeatures = 2000) # data is already normalized
hvgs <- VariableFeatures(train_obj)
# hvg_diff <- hvgs[!(hvgs %in% row.names(expr_mat))]
# if (length(hvg_diff) > 0){ # solve the problem that replacing "_" with "-"
#   print("Tune hvg list...")
#   hvgs <- c(hvgs[hvgs %in% row.names(expr_mat)], str_replace(hvg_diff, "-", "_"))
# }
expr_mat_hvg <- expr_mat[hvgs, ]
print(sprintf("HVG data shape (gene x cell): %d x %d", dim(expr_mat_hvg)[1], dim(expr_mat_hvg)[2]))

write.csv(t(as.matrix(expr_mat_hvg)), sprintf("./data/single_cell/experimental/mammalian_cerebral_cortex/new_processed/%s-norm_data-hvg.csv", split_type))
write.csv(hvgs, sprintf("./data/single_cell/experimental/mammalian_cerebral_cortex/new_processed/%s-var_genes_list.csv", split_type))
write.csv(meta_df, "./data/single_cell/experimental/mammalian_cerebral_cortex/new_processed/meta_data.csv")