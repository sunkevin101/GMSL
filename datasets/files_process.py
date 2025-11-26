import pandas as pd

from datasets.load_utils import get_overlapped_patient


def process_file_data(clinical_path, clinical_omics_folder, omic_modal, wsi_embedding_folder, n_bins=4, eps=1e-6):
    """
    Parse file data, read files, and obtain discrete labels
    Returns: df_clinical, df_gene, and patient_list
    Supports single-modal and multi-modal modes
    """

    # ====== Read Data ======
    # Read clinical info into df: .csv -> df_clinical
    df_clinical = pd.read_csv(clinical_path)
    df_clinical = df_clinical.dropna()  # Drop rows with any missing values

    # Process omic modalities into df_gene
    if omic_modal == 'Multi_modal':
        # Multi-modal: load three modalities
        # df_gene: dict 
        df_gene = {}
        
        # Load mRNA data
        mrna_path = '{}/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'.format(clinical_omics_folder)
        df_mrna = pd.read_csv(mrna_path, delimiter="\t")
        df_mrna = df_mrna.drop(['Entrez_Gene_Id'], axis=1)
        df_mrna = df_mrna[df_mrna['Hugo_Symbol'].notna()].dropna()
        df_mrna = df_mrna.set_index('Hugo_Symbol')
        df_gene['mRNA'] = df_mrna
        
        # Load CNA data
        cna_path = '{}/data_cna.txt'.format(clinical_omics_folder)
        df_cna = pd.read_csv(cna_path, delimiter="\t")
        df_cna = df_cna.drop(['Entrez_Gene_Id'], axis=1)
        df_cna = df_cna[df_cna['Hugo_Symbol'].notna()].dropna()
        df_cna = df_cna.set_index('Hugo_Symbol')
        df_gene['CNA'] = df_cna
        
        # Load methylation data
        methylation_path = '{}/data_methylation_hm27_hm450_merged.txt'.format(clinical_omics_folder)
        df_methylation = pd.read_csv(methylation_path, delimiter="\t")
        df_methylation = df_methylation.drop(['ENTITY_STABLE_ID'], axis=1)
        df_methylation = df_methylation.drop(['DESCRIPTION'], axis=1)
        df_methylation = df_methylation.drop(['TRANSCRIPT_ID'], axis=1)
        df_methylation = df_methylation[df_methylation['NAME'].notna()].dropna()
        df_methylation = df_methylation.set_index('NAME')
        df_gene['Methylation'] = df_methylation

    else:
        # Single-modal: keep original logic
        # df_gene: DataFrame

        # Process omic data path
        if omic_modal == 'mRNA':
            z_score_path = '{}/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'.format(clinical_omics_folder)
        elif omic_modal == 'CNA':
            z_score_path = '{}/data_cna.txt'.format(clinical_omics_folder)
        else:
            z_score_path = '{}/data_methylation_hm27_hm450_merged.txt'.format(clinical_omics_folder)

        # Process omic modality data into df
        if omic_modal == 'mRNA':
            df_mrna = pd.read_csv(z_score_path, delimiter="\t")
            df_mrna = df_mrna.drop(['Entrez_Gene_Id'], axis=1)
            df_mrna = df_mrna[df_mrna['Hugo_Symbol'].notna()].dropna()
            df_mrna = df_mrna.set_index('Hugo_Symbol')
            df_gene = df_mrna

        elif omic_modal == 'CNA':
            df_cna = pd.read_csv(z_score_path, delimiter="\t")
            # Drop ENTITY_STABLE_ID; remove rows with missing Hugo_Symbol; set Hugo_Symbol as index
            df_cna = df_cna.drop(['Entrez_Gene_Id'], axis=1)
            df_cna = df_cna[df_cna['Hugo_Symbol'].notna()].dropna()
            df_cna = df_cna.set_index('Hugo_Symbol')
            df_gene = df_cna

        else:
            df_methylation = pd.read_csv(z_score_path, delimiter="\t")
            df_methylation = df_methylation.drop(['ENTITY_STABLE_ID'], axis=1)
            df_methylation = df_methylation.drop(['DESCRIPTION'], axis=1)
            df_methylation = df_methylation.drop(['TRANSCRIPT_ID'], axis=1)
            df_methylation = df_methylation[df_methylation['NAME'].notna()].dropna()
            df_methylation = df_methylation.set_index('NAME')
            df_gene = df_methylation

    # After dropping missing rows, determine patient list via WSI, clinical, and omics; use first 12 chars
    patient_id_list = get_overlapped_patient(wsi_embedding_folder, df_clinical, df_gene)

    # Process df_clinical, only take deceased cases
    df_uncensored = df_clinical[df_clinical.OS_STATUS == '1:DECEASED']

    # Use OS_MONTHS to split into q_bins classes; q_bins as boundaries
    disc_labels, q_bins = pd.qcut(df_uncensored['OS_MONTHS'], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = df_clinical['OS_MONTHS'].max() + eps
    q_bins[0] = df_clinical['OS_MONTHS'].min() - eps

    # Split data into q_bins by boundaries and obtain labels
    disc_labels, q_bins = pd.cut(df_clinical['OS_MONTHS'], bins=q_bins,
                                 retbins=True, labels=False, right=False, include_lowest=True)
    df_clinical.insert(2, 'label', disc_labels.values.astype(int))  # insert label at column 2 in df_clinical

    return patient_id_list, df_clinical, df_gene
