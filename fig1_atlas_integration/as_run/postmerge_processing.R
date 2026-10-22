library(Seurat)
library(dplyr)
library(InSituType)
library(pheatmap)
library(yaml)
library(CellularNeighborhoods) #devtools::install_github("Nanostring-Biostats/CosMx-Analysis-Scratch-Space@Main", subdir = "_code/cellular-neighborhoods")

filename = "IBD_RNA"
organname = "IBD"

# Load the expression matrix and metadata for the samples (from h5ad processed)
expression_matrix <- read.csv(paste0(filename, "_raw_expression_matrix.csv"), row.names = 1)
#cell_metadata <- read.csv(paste0(filename, "_cell_metadata.csv"), row.names = 1)
cell_metadata <- read.csv(paste0(filename, "_updated_cell_metadata.csv"), row.names = 1)
gene_metadata <- read.csv(paste0(filename, "_gene_metadata.csv"), row.names = 1)
yaml_data <- yaml::read_yaml(paste0("ProjectConfig_", organname, ".yaml"))

# Ensure cell names match
if(!all(rownames(cell_metadata) %in% rownames(expression_matrix))) {
  stop("Mismatch between cell names in metadata and expression matrix")
}
# Reorder metadata to match the expression matrix cell order
cell_metadata <- cell_metadata[rownames(expression_matrix), ]

# QC as NS recommendations: 
# We generally require 20 counts per cell in 1000plex data and 50 counts in 6000plex data.
count_threshold <- 20
flag <- cell_metadata$nCount_RNA < count_threshold
table(flag)

# Then we’ll look for very large cells. 
# You can run a formal test for outliers on the vector of cell areas, or you can determine a reasonable threshold yourself:
hist(cell_metadata$Area, breaks = 100, xlab = "Cell Area", main = "")
# based on the above, set a threshold:
area_threshold <- 20000
abline(v = area_threshold, col = "red")
flag <- flag | (cell_metadata$Area > area_threshold)
table(flag)

### not run:
source("https://raw.githubusercontent.com/Nanostring-Biostats/CosMx-Analysis-Scratch-Space/Main/_code/FOV%20QC/FOV%20QC%20utils.R")
source("https://raw.githubusercontent.com/Nanostring-Biostats/CosMx-Analysis-Scratch-Space/Main/_code/vignette/utils/condenseTissues.R")
allbarcodes <- readRDS(url("https://github.com/Nanostring-Biostats/CosMx-Analysis-Scratch-Space/raw/Main/_code/FOV%20QC/barcodes_by_panel.RDS"))
xy <- as.matrix(cell_metadata[, c("CenterX_global_px", "CenterY_global_px")])
rownames(xy) <- rownames(cell_metadata)
thisinstrument_nanometers_per_pixel = 120.280945  
xy <- xy * thisinstrument_nanometers_per_pixel / 1000000
colnames(xy) <- paste0(c("x", "y"), "_mm")
xy <- condenseTissues(xy = xy, 
                      tissue = cell_metadata$Run_Tissue_name, 
                      tissueorder = NULL,  # optional, specify the order tissues are tiled in
                      buffer = 1, # space between tissues
                      widthheightratio = 1) # desired shape of final xy locations 
barcodemap <- allbarcodes$Hs_UCC
cell_metadata$SAHA_FOV <- paste0(cell_metadata$SAHA_batchcode, "_", cell_metadata$fov)

fovqcresult <- runFOVQC(counts = as.matrix(expression_matrix), xy = xy, fov = cell_metadata$SAHA_FOV, barcodemap = barcodemap, max_prop_loss = 0.3)
#mapFlaggedFOVs(fovqcresult)
#FOVSignalLossSpatialPlot(fovqcresult, shownames = TRUE)
flag <- flag | is.element(cell_metadata$fov, fovqcresult$flaggedfovs)
table(flag)

# to filter flagged outputs
#counts <- counts[!flag, ]
#negprobes_matrix <- negprobes_matrix[!flag, ]
#metadata <- cell_metadata[!flag, ]
#xy <- xy[!flag, ]

# Then create the Seurat object
#seurat_object <- CreateSeuratObject(counts = expression_matrix, meta.data = cell_metadata)


# Add gene metadata to the Seurat object if necessary
# Note: Seurat does not use gene metadata directly, but you can keep it for reference

source("../../../Code/Processing/Helpers/AtoMx_Sobj_Processing_helpers.R")
load("/Users/jiwoonpark/Dropbox (Personal)/2020- MasonLab/202202_SAHA/Data/Reference/NScellprofilelib/Gut_HCA.RData")
# or profile_matrix = read.csv("/Users/jiwoonpark/Dropbox (Personal)/2020- MasonLab/202202_SAHA/Data/Reference/Liver/CosMx_Liver.profiles.csv",row.names=1)

counts_matrix <- expression_matrix
#counts_matrix <- GetAssayData(seurat_object, assay = "RNA", slot = "counts") 
selected_features <- grep("^SystemControl\\d+", colnames(counts_matrix), value = TRUE)
negprobes_matrix <- counts_matrix[,selected_features]
counts <- counts_matrix %>% as.matrix()
#non_system_control <- !grepl("^SystemControl", colnames(counts))
#counts <- counts[,non_system_control]
negmean <- Matrix::rowMeans(as.matrix(negprobes_matrix))
cell_metadata$negmean <- negmean
cell_metadata$totalcounts <- mean(rowSums(counts))
negmean_per_totcount <- mean(negmean) / mean(rowSums(counts))
per_cell_bg <- rowSums(counts) * negmean_per_totcount

astats <- get_anchor_stats(counts = counts, neg = negmean, profiles = as.matrix(profile_matrix))
anchors <- choose_anchors_from_stats(counts = counts, neg = negmean, bg = per_cell_bg, anchorstats = astats, n_cells = 50, min_cosine = 0.4, min_scaled_llr = 0.03, insufficient_anchors_thresh = 5)
updatedprofiles <- updateReferenceProfiles(reference_profiles = profile_matrix, counts = counts, neg = negmean, bg = per_cell_bg, anchors = anchors)

immunofluordata <- matrix(rpois(n = nrow(counts) * 4, lambda = 100), nrow(counts))
cohort <- fastCohorting(immunofluordata, gaussian_transform = TRUE)
insitutype_res <- insitutype(x = counts, neg = negmean, cohort = cohort, bg = NULL, n_clusts = 5:10, reference_profiles = updatedprofiles$updated_profiles, update_reference_profiles = FALSE, n_phase1 = 200, n_phase2 = 500, n_phase3 = 2000, n_starts = 1, max_iters = 5)


### get metadata in: 
cell_metadata$SAHA_FOV <- paste0(filename, "_", cell_metadata$SAHA_batchcode, "_", cell_metadata$fov)
cell_metadata$SAHA_FOV <- paste0(filename, "_", cell_metadata$SAHA_batchcode, "_", cell_metadata$SAHA_withinsample_batch, "_", cell_metadata$fov)
fov_to_sample <- list()
for (tissue in names(yaml_data$FOV_maps)) {
  for (sample in names(yaml_data$FOV_maps[[tissue]])) {
    fovs <- yaml_data$FOV_maps[[tissue]][[sample]]
    for (fov in fovs) {
      fov_to_sample[[fov]] <- sample
    }
  }
}
fov_to_sample <- unlist(fov_to_sample)
cell_metadata <- cell_metadata %>% mutate(SAHA_Sample = fov_to_sample[SAHA_FOV])


## run insitutype
if (method == "semisupervised") {
  print("Annotate spatial data using semisupervised method")
  insitutype_res <- insitutype(x = counts, neg = negmean, cohort = cohort, bg = NULL, n_clusts = 5:10, reference_profiles = updatedprofiles$updated_profiles, update_reference_profiles = FALSE, n_phase1 = 200, n_phase2 = 500, n_phase3 = 2000, n_starts = 1, max_iters = 5)
  } else if (method == "supervised") {
    print("Annotate spatial data using supervised method")
    insitutype_res <- insitutypeML(x = t(counts), neg = negmean, reference_profiles = as.matrix(profile_matrix))
    } else {
      stop("incorrect method selected (semisupervised or supervised)", call. = FALSE)
    }

fp <- flightpath_plot(flightpath_result = NULL, insitutype_result = insitutype_res)

pheatmap::pheatmap(sweep(insitutype_res$profiles, 1, pmax(apply(insitutype_res$profiles, 1, max), .2), "/"), scale = "none", main = "Mean cell type expression profiles")

# Define a threshold for significance
threshold <- 0.5

# Calculate the mean expression for each row
row_means <- rowMeans(insitutype_res$profiles)

# Filter out rows where the name starts with "SystemControl"
non_system_control <- !grepl("^SystemControl", rownames(insitutype_res$profiles))

# Subset rows where the mean expression is greater than the threshold and names don't start with "SystemControl"
significant_profiles <- insitutype_res$profiles[row_means > threshold & non_system_control, ]

# Apply sweep and create the heatmap
pheatmap::pheatmap(sweep(significant_profiles, 1, pmax(apply(significant_profiles, 1, max), .2), "/"),
                   scale = "none", main = "Mean cell type expression profiles")

plot(xy, pch = 16, cex = 0.05, col = InSituType::colorCellTypes(freqs = table(insitutype_res$clust), palette = "brewers")[insitutype_res$clust], asp = 1)


newclusts <- refineClusters(
  logliks = insitutype_res$logliks,
  merges = c("a" = "epithelial-1-prostate", "b" = "stromal-prostate", "c" = "epithelial-2-prostate", "d" = "other-1-prostate", "e" = "other-2-prostate", "f" = "epithelial-3-prostate", "g" = "other-1-prostate", "h" = "epithelial-4-prostate", "i" = "stromal-prostate", "j" = "epithelial-5-prostate"),
  subcluster = NULL, # not recommended
  counts = counts,
  neg = negmean
) 
#COL_merges = c("a" = "epithelial-colon-01", "b" = "epithelial-colon-02", "c" = "epithelial-colon-03", "d" = "epithelial-colon-04", "e" = "epithelial-colon-01", "f" = "epithelial-colon-04", "g" = "immune-colon", "h" = "epithelial-colon-02", "i" = "epithelial-colon-01"),
#ILE_merges = c("a" = "epithelial-ileum-01", "b" = "pericyte-ileum-01", "c" = "epithelial-ileum-02", "d" = "immune-ileum", "e" = "immune-ileum", "f" = "pericyte-ileum-02", "g" = "plasma-ileum", "h" = "plasma-ileum", "i" = "immune-ileum", "j" = "epithelial-ileum-01")
#APE_merges = c("a" = "epithelial-appendix-01", "b" = "epithelial-appendix-02", "c" = "immune-appendix", "d" = "stem-appendix", "e" = "epithelial-appendix-04", "f" = "endothelial-appendix", "g" = "epithelial-appendix-02", "h" = "plasma-appendix")
#IBD_merges = c("a" = "plasma-ibd-01", "b" = "epithelial-ibd-01", "c" = "plasma-ibd-02", "d" = "epithelial-ibd-02", "e" = "epithelial-ibd-03", "f" = "epithelial-ibd-04", "g" = "epithelial-ibd-05", "h" = "stem-ibd")
#STO_merges = c("a" = "epithelial-stomach-01", "b" = "epithelial-stomach-02", "c" = "epithelial-stomach-03", "d" = "epithelial-stomach-04", "e" = "epithelial-stomach-05", "f" = "epithelial-stomach-06", "g" = "immune-stomach", "h" = "plasma-stomach", "i" = "epithelial-stomach-07", "j" = "epithelial-stomach-08")
#STO_HCA_merges = c("a" = "other-stomach", "b" = "immune-stomach-01", "c" = "epithelial-stomach-01", "d" = "immune-stomach-02", "e" = "epithelial-stomach-02", "f" = "other-stomach", "g" = "other-stomach", "h" = "epithelial-stomach-03", "i" = "epithelial-stomach-04", "j" = "epithelial-stomach-05")
#LIV_merges = c("a" = "other-liver", "b" = "immune-T-liver", "c" = "other-liver", "d" = "hepatocyte-01", "e" = "immune-liver", "f" = "hepatocyte-02", "g" = "other-liver", "h" = "other-liver", "i" = "hepatocyte-03", "j" = "other-liver")
#LIV_A2_merges = c("a" = "mix-liver-1", "b" = "erythroid-liver", "c" = "mix-liver-1", "d" = "mix-liver-1", "e" = "immune-liver-1", "f" = "mix-liver-1", "g" = "mix-liver-1", "h" = "immune-liver-1")
#PANC_merges = c("a" = "plasma-pancreas", "b" = "acinar-pancreas", "c" = "epithelial-pancreas", "d" = "mixedimmune-pancreas", "e" = "epithelial2-pancreas", "f" = "epithelial-pancreas", "g" = "ductal-pancreas")
#BM_merges = c("a" = "B-bonemarrow", "b" = "T-bonemarrow", "c" = "erythroids-bonemarrow", "d" = "other-bonemarrow", "e" = "plasma-bonemarrow", "f" = "other-bonemarrow", "g" = "erythroids-bonemarrow", "h" = "mixed", "i" = "plasma-bonemarrow")
#PROS_merges = c("a" = "epithelial-1-prostate", "b" = "stromal-prostate", "c" = "epithelial-2-prostate", "d" = "other-1-prostate", "e" = "other-2-prostate", "f" = "epithelial-3-prostate", "g" = "other-1-prostate", "h" = "epithelial-4-prostate", "i" = "stromal-prostate", "j" = "epithelial-5-prostate")


# Calculate the mean expression for each row
row_means <- rowMeans(newclusts$profiles)

# Filter out rows where the name starts with "SystemControl"
non_system_control <- !grepl("^SystemControl", rownames(newclusts$profiles))

# Subset rows where the mean expression is greater than the threshold and names don't start with "SystemControl"
significant_profiles <- newclusts$profiles[row_means > threshold & non_system_control, ]

# Apply sweep and create the heatmap
pheatmap::pheatmap(sweep(significant_profiles, 1, pmax(apply(significant_profiles, 1, max), .2), "/"),
                   scale = "none", main = "Mean cell type expression profiles")

saveRDS(insitutype_res, file = paste0(filename, "_insitutype_res.RDS"))

cell_metadata$Insitutype_Raw <- insitutype_res$clust
cell_metadata$Insitutype_Labelled <- newclusts$clust

write.table(table(cell_metadata$Insitutype_Labelled), file = "cellprop_label.csv", sep=",", col.names = TRUE, row.names = FALSE, quote = FALSE)

# get the broader type in

cellmaps <- read.csv("./cellprop_label.csv")
collapse_map <- setNames(cellmaps$Insitutype_Broad, cellmaps$Insitutype_Labelled)
cell_metadata$Insitutype_Broad <- factor(cell_metadata$Insitutype_Labelled, levels = names(collapse_map), labels = collapse_map)
cell_metadata$TissueCode <- filename
cell_metadata$cell_ID <- cell_metadata$cell
write.csv(cell_metadata, paste0(filename, "_updated_cell_metadata.csv"))  

# for napari metadata
# make sure the labels of interest is not your last column
cell_metadata_EC <- cell_metadata[cell_metadata$batch == "SAHA_CR_LIV_A1_TAP01",]
write.table(cell_metadata_EC, file="_metadata_EC.csv", sep=",", col.names=TRUE, row.names=FALSE, quote=FALSE)
write.table(cell_metadata, file="_metadata.csv", sep=",", col.names=TRUE, row.names=FALSE, quote=FALSE)


library(InSituCor)
non_system_control <- !grepl("^SystemControl", colnames(counts))
counts0 <- counts[,non_system_control]
conditionon <- cell_metadata[, c("SAHA_Sample", "SAHA_batchcode", "totalcounts", "negmean", "Insitutype_Labelled")]
rownames(conditionon) <- rownames(cell_metadata)
normcounts <- sweep(counts, 1, pmax(Matrix::rowSums(counts0), 1), "/")

res <- insitucor(
               # fundamental input data:
               counts = normcounts, conditionon = conditionon, celltype = conditionon$Insitutype_Labelled,
               # args for neighbor definition:
               neighbors = NULL, xy = xy, k = NULL, radius = 0.2, tissue = conditionon$SAHA_FOV, 
               # args for module definition:
               min_module_size = 3, max_module_size = 20, min_module_cor = 0.0001,
               gene_weighting_rule = "inverse_sqrt", resolution = 0.5,
               # args for controlling memory and compute
               roundcortozero = 0.05, max_cells = nrow(normcounts),
               # max_cells = 1e5, 
               # args for cell type attribution scoring
               attribution_subset_size = 5000, 
               verbose = TRUE)

# condition correlation matrix:
heatmap(as.matrix(res$condcor))

# each module with weights:
head(res$modules)

# single cell scores:
head(res$scores_sc)

# environment scores:
head(res$scores_env)

# plot one module's single cell and environmnt scores:
viridiscols <- c('#000004FF', '#1B0C42FF', '#4B0C6BFF', '#781C6DFF', '#A52C60FF', '#CF4446FF', '#ED6925FF', '#FB9A06FF', '#F7D03CFF', '#FCFFA4FF')
par(mfrow = c(1, 2))
plot(xy, asp = 1, pch = 16, cex = 0.05, cex.main = 0.75,
     main = paste0(colnames(scores$scores_env)[1], " single cell score"),
     xlab = "x mm", ylab = "y mm",
     col = viridiscols[1 + pmin(round(9 * (scores$scores_sc[, 1] / quantile(scores$scores_sc[, 1], 0.95))), 10)])
plot(xy, asp = 1, pch = 16, cex = 0.05, cex.main = 0.75,
     main = paste0(colnames(res$scores_env)[1], " environment score"),
     xlab = "x mm", ylab = "y mm",
     col = viridiscols[1 + pmin(round(9 * (res$scores_env[, 1] / quantile(res$scores_env[, 1], 0.99))), 10)])

# each cell type’s involvement in each module
heatmap(res$celltypeinvolvement, cexCol = .6, cexRow = .8, margins = c(8, 10))

# how each cell type contributes to each gene within a single module
str(res$attributionmats)
head(res$attributionmats[[1]])

# module gene content to identify interesting modules
plotCorrelationNetwork(res$condcor, modules = res$modules, show_gene_names = TRUE, corthresh = 0.1)


library(CellularNeighborhoods)
library(mclust)
library(spatstat)
# define neighbors using a K-nearest approach:
neighbors.nearest50 <- nearestNeighborGraph(x = xy[, 1], y = xy[, 2], N = 50)

# define using a radius-based approach:
neighbors.radiusbased <- radiusBasedGraph(x = xy[, 1], y = xy[, 2], R = 0.05)

# get matrix of mean expression in each cell's neighbors:
neighborhoodexpression <- neighbor_colMeans(x = normcounts, neighbors = neighbors.nearest50)

# reduce the matrix to its top PCs:
pc <- irlba::prcomp_irlba(neighborhoodexpression, n = 25)$x

# cluster the top PCs:
temp <- mclust::Mclust(pc, 
                       G = 10, # 6 clusters
                       modelNames = c("EEI"))
spatialclust <- temp$classification

# plot the results:
cols <- c("#616161","#4285f4","#db4437","#f4b400","#0f9d58","#ab47bc","#00acc1",
          "#ff7043","#9e9d24","#5c6bc0","#f06292","#00796b","#c2185b","#7e57c2",
          "#03a9f4","#8bc34a","#fdd835","#fb8c00","#8d6e63","#9e9e9e","#607d8b")
# plot:
sub = TRUE
plot(xy[sub, ], pch = 16, cex = 0.05, 
     asp = 1, xlab = "", ylab = "",
     col = cols[as.numeric(as.factor(spatialclust[sub]))])

pp <- spatstat.geom::ppp(xy[, 1], xy[, 2], xrange = range(xy[, 1]), yrange = range(xy[, 2]))
marks(pp) <- cell_metadata$Insitutype_Labelled
spatstat.geom::marks(pp) <- as.factor(spatstat.geom::marks(pp))
# count neighbors of each db cluster:
neighbormarks <- spatstat.explore::marktable(X = pp, R = NULL, N = 50, exclude=TRUE, collapse=FALSE)
neighbormarks <- as.matrix(neighbormarks)
rownames(neighbormarks) <- rownames(xy)
head(neighbormarks)

spatialclust <- mclust::Mclust(neighbormarks, #sweep(neighbormarks, 1, rowSums(neighbormarks), "/"),
                               G = 8, # 8 clusters
                               modelNames = c("EEI"))$classification
## plot the results:
# subset for faster plotting:
sub <- sample(1:nrow(xy), round(nrow(xy) / 20), replace = FALSE)
# color palette:
cols <- c("#616161","#4285f4","#db4437","#f4b400","#0f9d58","#ab47bc","#00acc1",
          "#ff7043","#9e9d24","#5c6bc0","#f06292","#00796b","#c2185b","#7e57c2",
          "#03a9f4","#8bc34a","#fdd835","#fb8c00","#8d6e63","#9e9e9e","#607d8b")
# plot:
plot(xy[sub, ], pch = 16, cex = 0.1, 
     asp = 1, 
     col = cols[as.numeric(as.factor(spatialclust[sub])) %% length(cols) + 1])

