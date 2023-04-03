# coding: utf-8
""" training procedure """
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score  # type: ignore
import torch  # type: ignore
from torch import sigmoid
from pytorchtools import EarlyStopping  # type: ignore
from src.mlflow_helpers import log_params_from_omegaconf_dict


def train_one_epoch(model,
                    device,
                    train_loader,
                    criterion,
                    optimizer) -> float:  # type: ignore
    """Make training for one epoch using all batches

    Parameters
    ----------
    model
    device
    train_loader
    criterion
    optimizer

    Returns
    -------
    float
        loss for this epoch
    """
    running_loss = 0.0
    step = 0
    model.train()
    for data in train_loader:
        data.to(device)  # Use GPU
        # ####### DEBUG for pytorch Dataset, not for PYG ########
        # real_graph_indices = data.graph_index.detach().to('cpu').numpy() - data.ptr.detach().to('cpu').numpy()[:-1]
        # ######################
        # Reset gradients
        optimizer.zero_grad()
        out = model(x=data.x.float(),
                    edge_index=data.edge_index,
                    edge_weight=data.edge_weight,
                    batch=data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y.float())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += loss.item()
        step += 1

    return running_loss / step


@torch.no_grad()
def test_one_epoch(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    step = 0
    batch_auc = []
    batch_accuracy = []

    for data in test_loader:
        data.to(device)  # Use GPU
        # ####### DEBUG for pytorch Dataset, not for PYG ########
        # real_graph_indices = data.graph_index.detach().to('cpu').numpy() - data.ptr.detach().to('cpu').numpy()[:-1]
        # ######################
        out = model(x=data.x.float(),
                    edge_index=data.edge_index,
                    edge_weight=data.edge_weight,
                    batch=data.batch)
        loss = criterion(out, data.y.float())
        running_loss += loss.item()
        step += 1

        pred = sigmoid(out)
        y_true = data.y.detach().to('cpu').numpy()
        y_pred = pred.detach().to('cpu').numpy()
        batch_accuracy.append(accuracy_score(y_true, y_pred.round()))
        batch_auc.append(roc_auc_score(y_true, y_pred))

    return running_loss / step, np.mean(batch_auc), np.mean(batch_accuracy)


def reset_model(model):
    # Reinitialize layers
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_valid_model(model,
                      device,
                      train_loader,
                      valid_loader,
                      criterion,
                      optimizer,
                      cfg,
                      mlflow_object: Optional = None) -> float:  # type: ignore
    print("Start training ...")
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(**cfg.training.stoping)
    n_epochs = cfg.training.max_epoch
    if mlflow_object is not None:
        # log param
        log_params_from_omegaconf_dict(cfg)
    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        train_loss = train_one_epoch(model=model,
                                     device=device,
                                     train_loader=train_loader,
                                     criterion=criterion,
                                     optimizer=optimizer)
        # record training loss
        avg_train_losses.append(train_loss)

        ######################
        # validate the model #
        ######################
        valid_loss, valid_auc, valid_accr = test_one_epoch(model=model,
                                                           device=device,
                                                           test_loader=valid_loader,
                                                           criterion=criterion)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        if mlflow_object is not None:
            mlflow_object.log_metric('train loss', train_loss, step=epoch)
            mlflow_object.log_metric('valid loss', valid_loss, step=epoch)
            mlflow_object.log_metric('valid auc', valid_auc, step=epoch)
            mlflow_object.log_metric('valid accuracy', valid_accr, step=epoch)

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
                     f'train_loss: {train_loss:.5f} '
                     f'valid_loss: {valid_loss:.5f} '
                     f'Valid ROC AUC {valid_auc:.5f}')

        print(print_msg)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        patience_counter = early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            if mlflow_object is not None:
                mlflow_object.log_param('Learned Epochs', epoch - patience_counter)
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(cfg.training.stoping['path']))
    t_loss_train, t_mean_auc_train, t_mean_accr_train = test_one_epoch(model, device, train_loader, criterion)
    t_loss, t_mean_auc, t_mean_accr = test_one_epoch(model, device, valid_loader, criterion)
    print(f'Final TRAIN loss: {t_loss_train}, AUC: {t_mean_auc_train}, accuracy: {t_mean_accr_train}\n'
          f'Final VALID loss: {t_loss}, AUC: {t_mean_auc}, accuracy: {t_mean_accr}')

    return t_loss


def test_model(model,
               device,
               test_loader,
               criterion,
               cfg):
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(cfg.training.stoping['path']))
    t_loss_test, t_mean_auc_test, t_mean_accr_test = test_one_epoch(model, device, test_loader, criterion)
    print(f'Final TEST loss: {t_loss_test}, AUC: {t_mean_auc_test}, accuracy: {t_mean_accr_test}')
