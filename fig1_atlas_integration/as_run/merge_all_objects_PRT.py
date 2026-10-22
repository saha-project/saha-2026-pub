import os
import pandas as pd
import numpy as np
import re
import anndata as ad
import squidpy as sq
import scanpy as sc
import harmonypy as hm
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import diags

# Use the current directory as the data directory
data_directory = os.getcwd()

def find_directories(directory, pattern):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and pattern in d]

# Function to find files matching patterns in a directory
def find_files(directory, patterns):
    flatfile_dir = os.path.join(directory, "flatFiles")
    files = {}
    for key, pattern in patterns.items():
        matched_files = []
        for root, _, filenames in os.walk(flatfile_dir):
            matched_files.extend([os.path.join(root, f) for f in filenames if pattern in f])
        if matched_files:
            files[key] = matched_files[0]  # If multiple matches, take the first one
        else:
            files[key] = None
    return files

# Function to clean column names
def clean_column_names(df):
    new_columns = {}
    for col in df.columns:
        # Remove random prefixes based on known patterns
        cleaned_col = re.sub(r'[a-f0-9]{8}\.[a-f0-9]{4}\.[a-f0-9]{4}\.[a-f0-9]{4}\.[a-f0-9]{12}_\d', '', col)
        new_columns[col] = cleaned_col
    df.rename(columns=new_columns, inplace=True)
    return df

# Function to inspect and clean files
def inspect_and_clean_files(files):
    cleaned_dataframes = {}
    for key, file in files.items():
        print(f"{key}: {file}")
        if file:
            try:
                df = pd.read_csv(file)
                df = clean_column_names(df)
                print(f"Structure of {key}:")
                print(df.head())
                print(df.info())
                cleaned_dataframes[key] = df
            except Exception as e:
                print(f"Error reading {key} file: {e}")
        else:
            print(f"{key} file not found.")
    return cleaned_dataframes

# Function to normalize the counts based on the total counts (NS recommendation)
def normalize_expression(metadata, exprmat):
    # Calculate the scaling factor
    scaling_factor = np.mean(metadata['nCount_RNA'])
    
    # Create the diagonal matrix with scaling factors
    scaling_factors = scaling_factor / metadata['nCount_RNA']
    
    # Create a sparse diagonal matrix
    norm_matrix = diags(scaling_factors, offsets=0)
    
    # Perform the matrix multiplication to normalize the expression matrix
    norm_exprmat = norm_matrix.dot(exprmat)
    
    return norm_exprmat

# Function to load data manually and create AnnData
def create_anndata(sample_name, data_dir):
    sample_dir = os.path.join(data_dir, sample_name)
    
    # File patterns
    file_patterns = {
        'exprmat': "exprMat_file.csv.gz",
        'metadata': "metadata_file.csv.gz",
        'polygon': "polygons.csv.gz", # polygon is also optional
        'fov_positions': "fov_positions_file.csv.gz",
        'tx_file': "tx_file.csv.gz"  # tx_file is optional
    }
    
    files = find_files(sample_dir, file_patterns)
    cleaned_data = inspect_and_clean_files(files)
    
    # Extract cleaned data
    exprmat = cleaned_data.get('exprmat')
    exprmat['SAHA_cellID'] = sample + '_' + exprmat['fov'].astype(str) + '_' + exprmat['cell_ID'].astype(str)
    exprmat = exprmat.set_index('SAHA_cellID')
    exprmat = exprmat.drop(columns=['fov', 'cell_ID'])
    metadata = cleaned_data.get('metadata')
    metadata['cell_ID'] = metadata['cell'].str.split('_').str[-1].astype(int)
    metadata['SAHA_cellID'] = sample + '_' + metadata['fov'].astype(str) + '_' + metadata['cell_ID'].astype(str)
    metadata = metadata.set_index('SAHA_cellID')
    polygon = cleaned_data.get('polygon')
    fov_positions = cleaned_data.get('fov_positions')
    tx_file = cleaned_data.get('tx_file')

    # Check if all required files are present
    if any([exprmat is None, metadata is None, fov_positions is None]):
        raise ValueError(f"One or more required files are missing for sample '{sample_name}'")

    common_index = metadata.index.intersection(exprmat.index)
    # to generate sparse matrix: 
    #raw = csr_matrix(exprmat.astype(pd.SparseDtype("float64",0)).sparse.to_coo())
        
    # Create the AnnData object
    adata = ad.AnnData(
        csr_matrix(exprmat.loc[common_index, :].values, dtype=np.float32),
        obs=metadata.loc[common_index, :],
    )
    adata.var_names = exprmat.columns

    adata.obs['SAHA_name'] = sample_name
    adata.obs['SAHA_organcode'] = sample_name.split('_')[2]
    adata.obs['SAHA_batchcode'] = sample_name.split('_')[4]
        
    # Store polygons, fov_positions, and tx_file as dictionaries in adata.uns
    adata.uns['fov_positions'] = fov_positions.to_dict(orient='list')
    if polygon is not None:
        adata.uns['polygon'] = polygon.to_dict(orient='list')
    if tx_file is not None:
        adata.uns['tx_file'] = tx_file.to_dict(orient='list')

    if 'x' in metadata.columns and 'y' in metadata.columns:
        coords = metadata[['x', 'y']].values
        pixel_size = 0.12028
        coords = coords.mul(pixel_size)
        adata.obsm['spatial'] = coords.to_numpy()

    if 'CenterX_global_px' in metadata.columns and 'CenterY_global_px' in metadata.columns:
        adata.obsm['spatial'] = metadata[['CenterX_global_px', 'CenterY_global_px']].values

    # Normalize the counts and add the normalized counts as a new layer
    normalized_counts = normalize_expression(adata.obs, adata.X)
    adata.layers['normalized'] = normalized_counts

    # Also save raw counts as a layer for logtransformation later
    adata.layers['raw'] = adata.X
    adata.X = sp.csr_matrix(adata.X)

    adata.strings_to_categoricals()

    return adata

# Get all directories for RNA (_CR_) and Protein (_CP_)
protein_sample_list = find_directories(data_directory, "_CP_")

# Create and save separate AnnData objects for protein samples
protein_adatas = []
for sample in protein_sample_list:
    try:
        adata = create_anndata(sample, data_directory)
        protein_adatas.append(adata)
        adata.write(f"{sample}_squidpy.h5ad")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error processing protein sample '{sample}': {e}")

if protein_adatas:
    protein_merged_adata = ad.concat(protein_adatas, join='outer', label='batch', keys=protein_sample_list)

    # Debug: Check merged metadata for NaN and duplicates
    print("Merged RNA Metadata NaN count before processing:")
    print(protein_merged_adata.obs.isna().sum())

    # Save raw integrated object
    for column in protein_merged_adata.obs.columns:
        if protein_merged_adata.obs[column].dtype == 'object':
            protein_merged_adata.obs[column] = protein_merged_adata.obs[column].fillna('NA').astype(str)

    # Verify data types
    print("Data types after conversion:")
    print(protein_merged_adata.obs.dtypes)

    # Save the AnnData object
    try:
        protein_merged_adata.write("merged_squidpy_protein_raw.h5ad")
        print("File successfully saved.")
    except Exception as e:
        print(f"Error saving file: {e}")    
    
    # Perform QC, normalization, and filtering for highly variable genes
    sc.pp.filter_cells(protein_merged_adata, min_counts=20)
    sc.pp.filter_cells(protein_merged_adata, min_genes=1)
    sc.pp.filter_genes(protein_merged_adata, min_cells=1)
    sc.pp.normalize_total(protein_merged_adata, target_sum=1e4)
    sc.pp.log1p(protein_merged_adata)
    #sc.pp.highly_variable_genes(protein_merged_adata, n_top_genes=50, subset=True)
    sc.pp.scale(protein_merged_adata, max_value=10)

    # Run Harmony batch correction on Protein data
    sc.tl.pca(protein_merged_adata, svd_solver='arpack')
    ho = hm.run_harmony(protein_merged_adata.obsm["X_pca"], protein_merged_adata.obs, 'batch')
    protein_merged_adata.obsm["X_harmony"] = ho.Z_corr.T

    # Save UMAP before and after Harmony for Protein
    sc.pp.neighbors(protein_merged_adata)
    sc.tl.umap(protein_merged_adata)
    protein_merged_adata.obsm["X_umap_before_harmony"] = protein_merged_adata.obsm["X_umap"]
    sc.pl.umap(protein_merged_adata, color=['batch'], title="Protein UMAP before Harmony", save='umap_protein_before_harmony.pdf')
    sc.pl.umap(protein_merged_adata, color=['batch'], title="Protein UMAP before Harmony", save='umap_protein_before_harmony.png')
    
    sc.pp.neighbors(protein_merged_adata, use_rep="X_harmony")
    sc.tl.umap(protein_merged_adata)
    protein_merged_adata.obsm["X_umap_after_harmony"] = protein_merged_adata.obsm["X_umap"]
    sc.pl.umap(protein_merged_adata, color=['batch'], title="Protein UMAP after Harmony", save='umap_protein_after_harmony.pdf')
    sc.pl.umap(protein_merged_adata, color=['batch'], title="Protein UMAP after Harmony", save='umap_protein_after_harmony.png')

    # Save the processed RNA object
    protein_merged_adata.write("merged_squidpy_protein_processed.h5ad")

else:
    print("No valid Protein AnnData objects were created.")

