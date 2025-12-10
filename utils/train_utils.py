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



