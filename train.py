import random

from datasets.dataset_survival import *
from datasets.files_process import process_file_data
from datasets.dataset_splits import split_dataset
from datasets.load_utils import load_gene_family_info

from utils.configs import parse_args, print_training_params, log_training_params
from utils import log_utils
from utils.train_utils import *

from models.network import *

from engine.finetune_engine import *
from engine.loss_utils import *
from engine.pretrain_engine import *

import warnings
import datetime

warnings.filterwarnings("ignore")


def main():
    params = parse_args()  # config parameters

    # Start timing
    start_time = time.time()

    # fetch CUDA, and initialize
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda_device
    torch.manual_seed(params.kfold_split_seed)
    torch.cuda.manual_seed(params.kfold_split_seed)
    np.random.seed(params.kfold_split_seed)
    random.seed(params.kfold_split_seed)


    # ====== Workdir, logger path, and model save path ======
    # Work directory
    work_dir = params.work_dir
    experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M")

    # Display parameters before training 
    print_training_params(params, experiment_id)

    # Experiment result folder
    experiment_save_folder = '{}/{}/{}-{}-{}_{}/'.format(work_dir,
                                                         params.cancer_type,
                                                         params.main_folder,
                                                         params.cancer_type,
                                                         params.omic_modal,
                                                         experiment_id)
    create_directory(experiment_save_folder)

    # Model checkpoints save path
    model_save_folder = experiment_save_folder + '/5fold_checkpoints'
    create_directory(model_save_folder)

    # Logger save path
    log_save_path = experiment_save_folder + 'logger.txt'

    log_utils.set_logger(log_save_path)
    
    # ====== Datasets path ======
    # Clinical file path
    clinical_omics_folder = f'{params.clinical_omics_folder}/{params.cancer_type}'
    clinical_path = f'{clinical_omics_folder}/data_clinical_all_clean.csv'

    # Process clinical and genomics files
    # patient_id_list: determined by WSI, clinical, and omics; first 12 chars only
    patient_id_list, df_clinical, df_gene = process_file_data(clinical_path=clinical_path,
                                                              clinical_omics_folder=clinical_omics_folder,
                                                              omic_modal=params.omic_modal,
                                                              wsi_embedding_folder=params.wsi_embedding_folder)

    k_fold = 5 
    patient_split_dict = split_dataset(patient_id_list, df_clinical, fold=k_fold)  # evenly split data
    patient_split_keys = sorted(patient_split_dict.keys())  # sort keys to ensure consistent order
    print("5-fold cross validation split!\n")

    # Load genes in gene families; used when building dataset; 8 genomics groups
    family_genes_dict = load_gene_family_info('gene_family')
    
    
    # Prepare to record C-index and build table
    best_c_index_list = []
    result_dict = {i: [] for i in range(k_fold)}  # record best epoch and best c-index

    # ====== k-fold cross validation ======
    for fold in range(k_fold):
        logging.info('[ Fold {}/{} ] {}'.format(fold + 1, k_fold, '-' * 66))

        # Split train and test
        val_key = patient_split_keys[fold]
        val_patient_id = patient_split_dict[val_key]
        train_keys = [key for key in patient_split_keys if key != val_key]
        train_patient_id = [patient_id for train_key in train_keys for patient_id in patient_split_dict[train_key]]

        # ====== Build dataset and get dataloader ======
        train_dataset = Survival_Dataset(train_patient_id, df_gene, df_clinical, family_genes_dict,
                                         params.wsi_embedding_folder)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=12)
        val_dataset = Survival_Dataset(val_patient_id, df_gene, df_clinical, family_genes_dict,
                                       params.wsi_embedding_folder)
        val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=12)

        # Show a sample from test_loader
        x_path, x_omics, censorship, survival_months, label, patient_id = next(iter(val_loader))

        # ====== Initialize model per fold ======
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GMSL(omic_sizes=[i.shape[1] for i in x_omics],
                     n_classes=params.n_bins,
                     fusion=params.fusion_type,
                     model_size='small',
                     mask_ratio=params.mask_ratio)

        model.to(device)
        
        pretrain_optimizer = optim.Adam(model.parameters(),
                                        lr=params.pretrain_lr,
                                        weight_decay=params.pretrain_wd)

        

        logging.info(f'Number of patients for model training : {len(train_patient_id)}')
        logging.info(f'Number of patients for model validation : {len(val_patient_id)}')

        

        # Pretrain the MAE model and evaluate ------------------------------------
        best_pretrain_epoch, best_pretrain_model = pretrain_and_validation(params.pretrain_epochs,
                                                                           model,
                                                                           [train_loader, val_loader],
                                                                           pretrain_optimizer,
                                                                           device,
                                                                           grad_accumulation_steps=params.gradient_accumulation_steps)

        print(f"Fold: {fold + 1}/{k_fold}, Finetune:")
        
        # fetch: Finetune loss function
        loss_fn = NLLSurvLoss(alpha=params.alpha_surv)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.wd)

        # Finetune the model and evaluate ------------------------------------
        best_test_c_index, best_epoch = train_and_test(fold,
                                                       params.epochs,
                                                       best_pretrain_model,
                                                       [train_loader, val_loader],
                                                       optimizer, loss_fn, device, model_save_folder,
                                                       grad_accumulation_steps=params.gradient_accumulation_steps,
                                                       save_model=True)

        logging.info(f'[ in fold {fold + 1}, {best_epoch} epoch ] The highest test c-index: {best_test_c_index:.4f}')

        # Timing per epoch
        epoch_time = time.time()
        epoch_hr = (epoch_time - start_time) / 3600
        print('By now, training time: {} hrs'.format(epoch_hr))

        # Record each fold results
        best_c_index_list.append(best_test_c_index)
        result_dict[fold].append(best_epoch)
        result_dict[fold].append(best_test_c_index)

    # ====== Training complete; display parameters ======
    # Record training parameters
    log_training_params(params, experiment_id)

    # Summarize best results per fold and build table
    print_result_as_table(result_dict, k_fold)

    logging.info('Mean C-index: {:.3f}'.format(sum(best_c_index_list) / len(best_c_index_list)))
    logging.info(f'Std: {np.std(best_c_index_list):.3f}')
    logging.info('finish!')

    # Stop timing and display duration
    stop_time = time.time()
    time_hours = (stop_time - start_time) / 3600
    logging.info(f'training time : {time_hours:.2f} hrs')


if __name__ == "__main__":
    main()
