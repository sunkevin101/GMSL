import numpy as np
from tqdm import tqdm
import os
import torch
import gc
from sksurv.metrics import concordance_index_censored
import time
import pandas as pd

import json
import logging


def train_and_test(fold, epochs, model, loader_list, optimizer, loss_fn, device, save_path,
                   grad_accumulation_steps=32,
                   save_model=False):
    
    best_test_c_index = float('-inf')
    best_epoch = 0

    # train:test=4:1
    train_loader, test_loader = loader_list

    for epoch in range(epochs):
        print()

        # train the models
        train_loss, train_c_index = train_loop(model, train_loader, loss_fn, optimizer, device,
                                               grad_accumulation_steps=grad_accumulation_steps)

        # test the models
        test_loss, test_c_index = test_loop(model, test_loader, loss_fn, device)

        logging.info(f'Epoch: {epoch}, train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}')
        logging.info(f'Epoch: {epoch}, train_c_index: {train_c_index:.4f}, test_c_index: {test_c_index:.4f}')

        if test_c_index > best_test_c_index:

            best_test_c_index = test_c_index
            best_epoch = epoch

            if save_model:
                # save the best models
                torch.save(
                    {'epoch': epoch, 'model_state_dict': model.state_dict()},
                    save_path + f'/fold_{fold}_checkpoint.pt'
                )
                print(">Save a new models!<")



    return best_test_c_index, best_epoch


# ====== train, test ======
def train_loop(model, loader, loss_fn, optimizer, device, grad_accumulation_steps=32):

    model.train()

    train_loss = 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):

        data_WSI = data_WSI.squeeze().to(device)
        data_omic = [i.squeeze().to(device) for i in data_omic]
        label = label.to(device)
        c = c.to(device)

        hazards, S = model(x_path=data_WSI, x_omic=data_omic)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()  
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss += loss_value

        loss = loss / grad_accumulation_steps
        loss.backward()

        del data_WSI, data_omic, loss, hazards, S
        torch.cuda.empty_cache()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)

    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]

    return train_loss, c_index


def test_loop(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):
        data_WSI = data_WSI.squeeze().to(device)
        data_omic = [i.squeeze().to(device) for i in data_omic] 
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            hazards, S = model(x_path=data_WSI, x_omic=data_omic)

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss += loss_value  

        del data_WSI, data_omic, loss, hazards, S
        torch.cuda.empty_cache()

    # calculate loss and error for epoch
    val_loss /= len(loader)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool),  # event occurred indicator; death is event
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]

    return val_loss, c_index
