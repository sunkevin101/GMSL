# UNI, BLCA, mRNA
cancer_type='BLCA'
wsi=/home/kevin/wsl_ubuntu2204/DATA/Pathology_data/CLAM/BLCA/BLCA_512_univ1_features/pt_files

python train.py \
--wsi_embedding_folder ${wsi} \
--cancer_type ${cancer_type} \
--omic_modal 'mRNA' \
--pretrain_epochs 12 \
--pretrain_lr 1e-4 \
--epochs 10 \
--lr 5e-4 \
--mask_ratio 0.5
