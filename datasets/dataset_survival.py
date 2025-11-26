import numpy as np
import pandas as pd
import torch

from datasets.load_utils import *
from torch.utils.data import Dataset, DataLoader


class Survival_Dataset(Dataset):
    def __init__(self, patient_id_list, df_gene, df_clinical, dict_family_genes, feature_folder):  

        self.patient_id_list = patient_id_list
        self.df_gene = df_gene
        self.df_clinical = df_clinical
        self.dict_family_genes = dict_family_genes
        self.feature_folder = feature_folder

    def __len__(self):
        return len(self.patient_id_list)

    def __getitem__(self, idx):
        patient_id = self.patient_id_list[idx]

        x_omics = load_genomics_z_score(self.df_gene, patient_id, self.dict_family_genes)
        
        # Load feature now supports multi-slide processing
        x_path = load_feature(self.feature_folder, patient_id)

        survival_months, censorship, label = load_clinical(self.df_clinical, patient_id)

        return x_path, x_omics, censorship, survival_months, label, patient_id
