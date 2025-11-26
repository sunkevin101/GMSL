# GMSL, univ1, UCEC，mRNA
cancer_type='UCEC'
wsi='/home/kevin/wsl_ubuntu2204/DATA/Pathology_data/CLAM/UCEC/UCEC_512_univ1_1024d/pt_files'
genomics_and_labels='/mnt/e/Research/Pathology_Data/Genomics-data/self_omics'

python train.py \
--wsi_embedding_folder ${wsi} \
--clinical_omics_folder ${genomics_and_labels} \
--cancer_type ${cancer_type} \
--omic_modal 'mRNA' \
--pretrain_epochs 12 \
--pretrain_lr 1e-4 \
--epochs 10 \
--lr 5e-4 \
--mask_ratio 0.5 \
--delay_hours 0

