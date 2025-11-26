import collections
import os
import numpy as np
import pandas as pd
import torch



# Load genomics group dictionary
# From the specified folder path (gene_family_folder_path), load gene family information.
# Read all files under gene_family_folder_path and organize into a dictionary
# where keys are filenames (without extensions) and values are lists of lines in the files.
def load_gene_family_info(gene_family_folder_path):
    dir_list = os.listdir(gene_family_folder_path)
    dir_list = [gene_family_folder_path + '/' + i for i in dir_list]
    
    all_gene = []
    dict_family_genes = {}
    for p in dir_list:
        my_file = open(p, "r")
        data = my_file.read()

        # replacing end splitting the text when newline '\n' is seen.
        data_into_list = data.split("\n")
        all_gene += data_into_list
        gene_family = p.split('/')[-1].split('.')[0]

        dict_family_genes[gene_family] = data_into_list
        my_file.close()

    return dict_family_genes


def load_feature(feature_folder, patient_id):
    # patient_id has 12 chars, find all .pt files whose first 12 chars match
    path_features = []
    
    for filename in os.listdir(feature_folder):
        if filename.endswith('.pt') and filename.startswith(patient_id):
            file_path = os.path.join(feature_folder, filename)
            try:
                feature = torch.load(file_path)
                path_features.append(feature)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue
    # Can also consider OOM as well
    
    # concatenate all features
    concatenated_features = torch.cat(path_features, dim=0)
    return concatenated_features




# Load genomics data
def load_genomics_z_score(df_gene, patient_id, dict_family_genes):
    """
    Load genomics data supporting single-modal and multi-modal modes
    """
    x_omics = []
    
    patient_id = patient_id + '-01'  # append '-01'
    
    if isinstance(df_gene, dict):
        # Multi-modal: load three modalities per gene group
        # df_gene: dict data
        for _, gene_list in dict_family_genes.items():
            family_omics = []
            
            for _, df_gene_modal in df_gene.items():
                genes_modal = list(df_gene_modal.index)

                # In gene_list, find genes present in genes_modal and without ';' to form modal_gene_list
                modal_gene_list = [i for i in gene_list if i in genes_modal]
                modal_gene_list = [i for i in modal_gene_list if ';' not in i]

                # Take values for patient_id at modal_gene_list from df_gene_modal; remove NaN and missing
                if modal_gene_list:  
                    # If modal_gene_list is not empty, proceed with loading data
                    gene_z_scores = df_gene_modal[patient_id].loc[modal_gene_list]
                    gene_z_scores = gene_z_scores[gene_z_scores.notna()].dropna()

                    if len(gene_z_scores) > 0:
                        # Convert single gene modality data to tensor and append to family_omics
                        family_omics.append(torch.tensor(np.array(gene_z_scores).astype(np.float32)))
                    else:   
                        family_omics.append(torch.tensor([]).float())  # create empty tensor if no valid data
                else:
                    family_omics.append(torch.tensor([]).float()) # create empty tensor if no matching genes
            

            # Merge three modalities into one tensor
            if all(len(omic) > 0 for omic in family_omics):
                # If all modalities have data, concatenate
                combined_omics = torch.cat(family_omics, dim=0)
                x_omics.append(combined_omics)
            else:
                # If some modalities lack data, use available ones
                valid_omics = [omic for omic in family_omics if len(omic) > 0]
                if valid_omics:
                    combined_omics = torch.cat(valid_omics, dim=0)
                    x_omics.append(combined_omics)
                else:
                    x_omics.append(torch.tensor([]).float())  # create empty tensor if none have data
    else:
        # Single-modal: directly read df_gene
        for family, gene_list in dict_family_genes.items():
            genes = list(df_gene.index)

            # In gene_list, find genes present in genes and without ';'
            gene_list = [i for i in gene_list if i in genes]
            gene_list = [i for i in gene_list if ';' not in i]

            # Take values for patient_id at gene_list from df_gene; remove NaN and missing; append to x_omic
            gene_z_scores = df_gene[patient_id].loc[gene_list]
            gene_z_scores = gene_z_scores[gene_z_scores.notna()].dropna()
            x_omics.append(torch.tensor(np.array(gene_z_scores).astype(np.float32)))

    return x_omics



# Load patient clinical info
def load_clinical(df_clinical, patient_id):
    ''' Kexin have upadated the code about censorship '''
    df_patient = df_clinical[df_clinical.PATIENT_ID == patient_id[0:12]]
    censorship = df_patient.OS_STATUS.values[0] # deceased=0, living=1
    survival_event = df_patient.OS_MONTHS.values[0]
    label = df_patient.label.values[0]  # positively related to OS_MONTHS; specified in main (4 classes)

    return survival_event, 0 if censorship == '1:DECEASED' else 1, label



# Remove duplicates, Get patient list
# Filtering conditions:
# 1. Have genomics information (cbioportal)
# 2. Have WSI (TCGA)
# 3. Have clinical information (TCGA)
def get_overlapped_patient(path_img, df_clinical, df_gene):
    """
    Get overlapped patient list supporting single and multi-modal
    clinical has 12 chars
    pt and omics have 15 chars with '-01'
    """
    patient_list = []

    # WSI
    # Take first 15 chars; use set to deduplicate
    patient_img = os.listdir(path_img)
    patient_img_15 = list(set([i[0:15] for i in patient_img]))


    # genomics data
    # Take 15 chars; deduplicate
    if isinstance(df_gene, dict):
        # If df_gene is dict, multi-modal case
        # Multi-modal: require patients having all three modalities       
        patient_genomics_15 = []
        for _, df_modal in df_gene.items():
            modal_patients_15 = list(set([i[0:15] for i in list(df_modal.columns)[2:]]))
            if not patient_genomics_15:
                # If patient_genomics is empty, assign directly
                patient_genomics_15 = modal_patients_15
            else:
                # Take intersection to ensure all modalities have data
                patient_genomics_15 = [i for i in patient_genomics_15 if i in modal_patients_15]
        patient_list_15 = [i for i in patient_img_15 if i in patient_genomics_15]

    else:
        # Single-modal: deduplicate directly
        patient_genomic_15 = list(set([i[0:15] for i in list(df_gene.columns)[2:]]))
        patient_list_15 = [i for i in patient_img_15 if i in patient_genomic_15]


    # clinical information
    # 12 chars; deduplicate with set
    patient_clinical = list(set(list(df_clinical.PATIENT_ID)))
    patient_list = [i[0:12] for i in patient_list_15 if i[0:12] in patient_clinical]

    return patient_list
