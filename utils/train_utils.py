# ====== Delayed Run ======
from tqdm import tqdm
import time

def run_after_hours(delay_hours):
    delay_minutes = int(delay_hours * 60)
    print(f"\nrun the code after {delay_minutes} minutes")
    if delay_minutes > 0:
        for t in tqdm(range(delay_minutes)):
            time.sleep(60)  # 1 min = 60 s


# ====== Create New Directory ======
import os
def create_directory(save_path):
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)



# ====== Result Table ======
def print_result_as_table(result_dict, num_fold):
    
    print('+-------+-------------+--------------+')
    print('|  Fold | Best epoch  | Best c-index |')
    print('+-------+-------------+--------------+')
    for i in range(num_fold):
        print('|   {}   |      {}      |    {:.4f}    |'.format(i + 1, result_dict[i][0], result_dict[i][1]))
        print('+-------+-------------+--------------+')


def print_result_as_table_finetune_model(result_dict, num_fold):
    print('+-------+--------------+')
    print('|  Fold | Best c-index |')
    print('+-------+--------------+')
    for i in range(num_fold):
        print('|   {}   |    {:.4f}    |'.format(i + 1, result_dict[i][0]))
        print('+-------+--------------+')

