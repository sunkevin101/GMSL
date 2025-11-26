import numpy as np
from tqdm import tqdm
import torch
import gc

import logging
import time
from sksurv.metrics import concordance_index_censored


def pretrain_and_validation(epochs, model, loader_list, optimizer, device, grad_accumulation_steps=32):
    # pretrain pretext task
    best_epoch = 0
    min_train_loss = float('inf')
    best_pretrain_model = None

    # train:test=4:1
    train_loader, _ = loader_list

    for epoch in range(epochs):
        print()

        # train the MAE model; do not validate on test_loader
        train_loss = mae_train_loop(model, train_loader, optimizer, device, grad_accumulation_steps=grad_accumulation_steps)
        
        logging.info(f'Epoch: {epoch}, train_loss: {train_loss:.5f}')

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            best_epoch = epoch
            best_pretrain_model = model

    return best_epoch, best_pretrain_model


# ====== train, test ======
def mae_train_loop(model, loader, optimizer, device, grad_accumulation_steps=32):
    model.train()
    train_loss = 0.

    for batch_idx, (data_WSI, data_omic, c, event_time, label, patient_id) in enumerate(tqdm(loader)):
        data_WSI = data_WSI.squeeze().to(device)
        data_omic = [i.squeeze().to(device) for i in data_omic]

        loss = model(x_path=data_WSI, x_omic=data_omic, mode='pretrain')
        loss_value = loss.item()
        train_loss += loss_value

        loss = loss / grad_accumulation_steps
        loss.backward()

        del data_WSI, data_omic, loss
        torch.cuda.empty_cache()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)

    return train_loss
