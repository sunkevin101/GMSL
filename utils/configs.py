import argparse

import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')

    # Training dataset paths
    parser.add_argument('--cancer_type',
                        default='BRCA',
                        choices=['UCEC', 'BRCA', 'BLCA', 'LUAD', 'GBMLGG'],
                        help='cancer type used to delete the patient ID')
    parser.add_argument('--wsi_embedding_folder',
                        default='/home/kevin/SunKevin/CODE/ResNet50_pt/UCEC_pt_512_1024d_90_95_241107',
                        help='extracted feature folder')
    parser.add_argument('--clinical_omics_folder',
                        default='/mnt/e/A-Kevin/Research/Pathology_Data/Genomics-data/self_omics',
                        help='clinical and genomics folder, all in this folder')
    parser.add_argument('--omic_modal',
                        default='mRNA',
                        choices=['mRNA', 'CNA', 'Methylation', 'Multi_modal'],
                        type=str,
                        help='genomic type: single modal or multi_modal for all three modalities (mRNA, CNA, Methylation)')
    parser.add_argument('--k_fold', default=5, type=int, help='5-fold cross validation')

    # checkpoints save path
    parser.add_argument('--work_dir', default='./work_dir/GMSL', help='models weight and log save folder')
    parser.add_argument('--main_folder', default='gmsl-main', help='models folder name')

    # pretrain and finetune schedule
    parser.add_argument('--pretrain_epochs',
                        default=12,
                        type=int,
                        help='pretrain training epochs')
    parser.add_argument('--pretrain_lr',
                        default=1e-4,
                        type=float,
                        help='pretrain learning rate')
    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        help='finetune training epochs')
    parser.add_argument('--lr',
                        default=5e-4,
                        type=float,
                        help='finetune learning rate')

    parser.add_argument('--pretrain_wd', default=0, type=float, help='pretrain weight decay')
    parser.add_argument('--wd', default=0, type=float, help='finetune weight decay')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='default: 1, due to varying bag sizes, which is unallowed in dataloader')
    parser.add_argument('-gc', '--gradient_accumulation_steps',
                        default=32,
                        type=int,
                        help='Gradient backward Accumulation Step, in fact as batch_size in backward')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='WSI pretext task mask ratio')

    # finetune models
    parser.add_argument('--fusion_type', default='concat', choices=['concat', 'bilinear', 'images'],
                        help='models late fusion type')
    parser.add_argument('--alpha_surv', default=0.0, type=float,
                        help='How much to weigh uncensored patients, used in finetune')
    parser.add_argument('--n_bins', default=4, type=int, help='n survival event intervals')

    

    parser.add_argument('--cuda_device', default='0', type=str, help='gpu name')
    parser.add_argument('--kfold_split_seed', default=42, type=int, help='random seed')


    args = parser.parse_args()

    return args

N_const=88
def print_training_params(params,experiment_id):
    """
    Display all important parameters before training starts (using print)
    """

    print()
    print("Training Parameters:")
    print("=" * N_const)
    print(f"Cancer Type: {params.cancer_type}")
    if params.omic_modal == 'multi_modal':
        print(f"Omic Modal: multi_modal (mRNA, CNA, Methylation)")
    else:
        print(f"Omic Modal: {params.omic_modal}")
    
    print(f"Batch Size: {params.batch_size}")
    print(f"Gradient Accumulation Steps: {params.gradient_accumulation_steps}")
    print(f"Pretrain Epochs: {params.pretrain_epochs}")
    print(f"Pretrain Learning Rate: {params.pretrain_lr}")
    print(f"Mask Ratio: {params.mask_ratio}")
    print(f"Finetune Epochs: {params.epochs}")
    print(f"Finetune Learning Rate: {params.lr}")
    print(f"Fusion Type: {params.fusion_type}")

    print(f"Pretrain Weight Decay: {params.pretrain_wd}")
    print(f"Finetune Weight Decay: {params.wd}")
    print(f"Survival Loss Alpha: {params.alpha_surv}")
    print(f"K-fold Split Seed: {params.kfold_split_seed}")
    
    print(f"Clinical Omics Folder: {params.clinical_omics_folder}")
    print(f"WSI Embedding Folder: {params.wsi_embedding_folder}")
    print(f"Work Directory: {params.work_dir}")
    print(f"Main Folder: {params.main_folder}")
    print(f"Time ID: {experiment_id}")

    print("=" * N_const)
    print()


def log_training_params(params, experiment_id):
    """
    Log all parameters after training completion (using logging.info)
    """

    logging.info(" ")
    logging.info("Parameters Summary:")
    logging.info("=" * N_const)

    logging.info(f"Cancer Type: {params.cancer_type}")
    if params.omic_modal == 'multi_modal':
        logging.info(f"Omic Modal: multi_modal (mRNA, CNA, Methylation)")
    else:
        logging.info(f"Omic Modal: {params.omic_modal}")

    logging.info(f"Batch Size: {params.batch_size}")
    logging.info(f"Gradient Accumulation Steps: {params.gradient_accumulation_steps}")
    logging.info(f"Pretrain Epochs: {params.pretrain_epochs}")
    logging.info(f"Pretrain Learning Rate: {params.pretrain_lr}")
    logging.info(f"Mask Ratio: {params.mask_ratio}")
    logging.info(f"Finetune Epochs: {params.epochs}")
    logging.info(f"Finetune Learning Rate: {params.lr}")
    logging.info(f"Fusion Type: {params.fusion_type}")

    logging.info(f"Pretrain Weight Decay: {params.pretrain_wd}")
    logging.info(f"Finetune Weight Decay: {params.wd}")
    logging.info(f"Survival Loss Alpha: {params.alpha_surv}")
    logging.info(f"K-fold Split Seed: {params.kfold_split_seed}")
    
    logging.info(f"Clinical Omics Folder: {params.clinical_omics_folder}")
    logging.info(f"WSI Embedding Folder: {params.wsi_embedding_folder}")
    logging.info(f"Work Directory: {params.work_dir}")
    logging.info(f"Main Folder: {params.main_folder}")
    logging.info(f"Time ID: {experiment_id}")
    
    logging.info("=" * N_const)

