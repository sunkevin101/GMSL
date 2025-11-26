# Evenly split dataset

def split_dataset(patient_id_list, df_clinical, fold=5):
    # check if PATIENT_ID is in patient_id_list
    df_filtered = df_clinical[df_clinical['PATIENT_ID'].isin(patient_id_list)].copy()

    # split df_filtered into two parts: deceased and living
    deceased_df = df_filtered[df_filtered['OS_STATUS'] == '1:DECEASED'].sort_values(by='OS_MONTHS')
    living_df = df_filtered[df_filtered['OS_STATUS'] == '0:LIVING'].sort_values(by='OS_MONTHS')

    folds = {f'fold_{i}': [] for i in range(fold)}

    # split deceased_df into 5 folds
    for idx, row in enumerate(deceased_df.itertuples(), start=0):
        fold_idx = idx % fold
        folds[f'fold_{fold_idx}'].append(row.PATIENT_ID)

    # split living_df into 5 folds
    for idx, row in enumerate(living_df.itertuples(), start=0):
        fold_idx = idx % fold
        folds[f'fold_{fold_idx}'].append(row.PATIENT_ID)

    return folds


