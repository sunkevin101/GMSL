import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored



def test_model(model, loader, device):
    # Evaluate model performance
    model.eval()
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_patient_ids = []

    with torch.no_grad():
        for batch_idx, (data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):
            data_WSI = data_WSI.squeeze().to(device)
            data_omic = [i.squeeze().to(device) for i in data_omic]
            
            hazards, S = model(x_path=data_WSI, x_omic=data_omic)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            all_patient_ids.append(patient_id)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08)[0]

    return c_index, all_risk_scores, all_event_times, all_censorships, all_patient_ids