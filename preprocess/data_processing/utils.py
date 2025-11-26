import os

def create_directory(save_path):
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)


import pandas as pd
def get_overlapped_patient(path_img, df_clinical, df_gene):
    patient_list = []

    # WSI
    patient_img = os.listdir(path_img)
    patient_img = list(set([i[0:12] for i in patient_img]))

    # clinical information
    patient_clinical = list(set(list(df_clinical.PATIENT_ID)))
    patient_list = [i for i in patient_img if i in patient_clinical]

    # genomics data
    patient_genmonic = list(set([i[0:12] for i in list(df_gene.columns)[2:]]))
    patient_list = [i for i in patient_list if i in patient_genmonic]
    return patient_list


import time
def decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time:{(end_time - start_time) * 1000:.2f}ms")
        return result

    return wrapper