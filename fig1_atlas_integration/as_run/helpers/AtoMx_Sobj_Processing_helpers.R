### renaming RNA_normalized and other assay/column names so that it does not contain random long task ID
rename_atomx_norm_assayname <- function(Sobj = RNA) {
  ## rename assays
  ## if there is nbclustering run, change names
  if (unique(Sobj@meta.data$assay_type) == "RNA") {
    name01 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), "^spatialclust_.*assignments$")]
    name02 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), "^nn_.*_1$")]
    name03 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), "^RNA_nbclust_.*_1_clusters$")]
    name04 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), "^RNA_nbclust_.*_1_posterior_probability$")]
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name01] <- "spatialcluster_assignments"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name02] <- "nn_cluster_1"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name03] <- "RNA_nbclust_1_clusters"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name04] <- "RNA_nbclust_1_posterior_probability"
  } else {
    name01 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), "^spatialclust_.*assignments$")]
    name02 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), "^nn_.*_1$")]
    name03 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), ".*cell_type_number$")]
    name04 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), ".*final_cell_type$")]
    name05 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), ".*round_1$")]
    name06 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), ".*round_2$")]
    name07 <- colnames(Sobj@meta.data)[str_detect(colnames(Sobj@meta.data), ".*round_3$")]
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name01] <- "spatialcluster_assignments"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name02] <- "nn_cluster_1"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name03] <- "cell_type_number"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name04] <- "final_cell_type"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name05] <- "round_1"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name06] <- "round_2"
    colnames(Sobj@meta.data)[colnames(Sobj@meta.data) == name07] <- "round_3"
  }
  dynamic_assay_names <- names(Sobj@assays)
  pattern <- "^RNA_normalized_"
  dynamic_assay_name <- dynamic_assay_names[str_detect(dynamic_assay_names, pattern)]
  ## Check if we found exactly one match and rename it
  if (length(dynamic_assay_name) == 1) {
    new_assay_name <- "RNA_normalized"
    names(Sobj@assays)[names(Sobj@assays) == dynamic_assay_name] <- new_assay_name
    } else {
    warning("Either no matching assay was found or multiple matches exist. Please check the assay names.")
  }
  return(Sobj)
}



### Mapping sample IDs to FOVs
## Assuming "project" is your loaded YAML structure: e.g., project <- yaml::yaml.load_file('../../../Code/ProjectConfigs/ProjectConfig_APE.yaml')
map_sampleID_to_fovs <- function(sobj = RNA, yamlfile = project, fovname_column = "SAHA_FOV") {
  ## Initialize an empty vector for the mapping
  mapping <- c()
  ## Iterate through the YAML structure to fill the mapping
  for (category in names(yamlfile$FOV_maps)) {
    for (group in names(yamlfile$FOV_maps[[category]])) {
      for (fov in yamlfile$FOV_maps[[category]][[group]]) {
        mapping[fov] <- group
      }
    }
  }
  ## Assuming your data frame is named df and the column with FOV labels is named FOV_label
  ## Ensure that fovname_column is indeed a column in the mapping
  if (!fovname_column %in% colnames(sobj@meta.data)) {
    stop("The specified fovname_column does not exist in the seurat object", call. = FALSE)
  }
  ## Add a new column to the data frame with the group labels
  sobj@meta.data <- sobj@meta.data %>%  mutate(Sample_ID = mapping[as.character(sobj@meta.data[[fovname_column]])])
  return(sobj)
}



### RNA cell typing using insitutype
## Automated annotation of spatial data with single-cell references using InSituType
## Assumes `ref` is a seurat object with a "cell_type" metadata column--can be overridden
## NOTE: originally, annotateData() removed cells with <10 transcripts, so be cautious of the data composition changing slightly
## changed to 1
annotate_RNA_insitu_data <- function(Sobj, ref, method = "supervised") {
  ## load ref profile matrix from project yaml
  load(project$paths$ref_celltype_directory)

  ## filter out cells with <1 transcripts
  cells_keep <- colnames(Sobj)[colSums(Sobj[["RNA"]]$counts) > 1]
  Sobj <- subset(Sobj, cells = cells_keep)
  query_mat <- Sobj[["RNA"]]$counts

  ## prepare for insitutype inputs
  negmean <- colMeans(Sobj[["negprobes"]]$counts)
  negmean_per_totcount <- mean(negmean) / mean(rowSums(query_mat))
  per_cell_bg <- rowSums(query_mat) * negmean_per_totcount
  astats <- get_anchor_stats(counts = query_mat, neg = negmean, profiles = as.matrix(profile_matrix))

  ## choose anchors and update ref profile based on the results
  anchors <- choose_anchors_from_stats(counts = query_mat, neg = negmean, bg = per_cell_bg, anchorstats = astats, n_cells = 50, min_cosine = 0.4, min_scaled_llr = 0.03, insufficient_anchors_thresh = 5)
  updatedprofiles <- updateReferenceProfiles(reference_profiles = profile_matrix, counts = query_mat, neg = negmean, bg = per_cell_bg, anchors = anchors)

  ## automatic cohorting, which is a low-risk undertaking: 
  immunofluordata <- matrix(rpois(n = nrow(query_mat) * 4, lambda = 100), nrow(query_mat))
  cohort <- fastCohorting(immunofluordata, gaussian_transform = TRUE)
  ## run insitutype
  if (method == "semisupervised") {
    print("Annotate spatial data using semisupervised method")
    insitutype_res <- insitutype(x = query_mat, neg = negmean, cohort = cohort, bg = NULL, n_clusts = 5:10, reference_profiles = updatedprofiles$updated_profiles, update_reference_profiles = FALSE, n_phase1 = 200, n_phase2 = 500, n_phase3 = 2000, n_starts = 1, max_iters = 5)
    } else if (method == "supervised") {
      print("Annotate spatial data using supervised method")
      insitutype_res <- insitutypeML(x = t(query_mat), neg = negmean, reference_profiles = as.matrix(profile_matrix))
      } else {
        stop("incorrect method selected (semisupervised or supervised)", call. = FALSE)
      }

  Sobj$celltype_pred <- insitutype_res$clust
  return(Sobj)
}

annotate_PRT_insitu_data <- function(Sobj, ref) {
  ## load ref profile matrix from project yaml
  prior_marker_info <- read.csv(project$paths$ref_celesta_directory)
  query_mat <- Sobj[["RNA"]]$counts
  imaging_data <- t(query_mat) %>% as.data.frame()
  #imaging_data <- as.data.frame(t(as.matrix(Sobj@assays$RNA$counts)))
  #imaging_data$celltype <- Sobj$cell_type_number
  imaging_data$X <- Sobj$x_slide_mm
  imaging_data$Y <- Sobj$y_slide_mm
  overlapping_markers <- colnames(prior_marker_info)[colnames(prior_marker_info) %in% colnames(imaging_data)]
  CelestaObj <- CreateCelestaObject(project_title = "CosMx_protein", prior_marker_info[,c("celltype", "Lineage_level", overlapping_markers)], imaging_data[,c("X", "Y", overlapping_markers)])
  CelestaObj <- FilterCells(CelestaObj, high_marker_threshold = 0.9, low_marker_threshold = 0.4)
  CelestaObj <- AssignCells(CelestaObj, max_iteration = 10, cell_change_threshold = 0.01,
                          high_expression_threshold_anchor = high_marker_threshold_anchor,
                          low_expression_threshold_anchor = low_marker_threshold_anchor,
                          high_expression_threshold_index = high_marker_threshold_iteration,
                          low_expression_threshold_index = low_marker_threshold_iteration)
  return(CelestaObj)
}
#Sobj$celltype_celesta <- CelestaObj@final_cell_type_assignment






