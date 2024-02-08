#  +-+ +-+ +-+ +-+ +-+ +-+
#  |i| |m| |p| |o| |r| |t|
#  +-+ +-+ +-+ +-+ +-+ +-+

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

import pandas as pd

import os

from training import train, evaluate
from networks import pretrained_network
from datasets import ImageDataset

#  +-+ +-+ +-+ +-+
#  |m| |a| |i| |n|
#  +-+ +-+ +-+ +-+


def main(exp, dataset_name, model_name, train_opt=True, eval_opt=True, save=True, overwrite=True, batch_size=16, image_size=256, 
         learning_rate=0.0001, epochs=100, printing=True, return_acc=False, 
         checkpoint_path='./results/checkpoints', log_path='./results/log', evals_path='./results/evals', comet=None):
    """
    Trains and evaluates a deep learning model on a given dataset.

    Args:
        dataset_name (str): The name of the dataset to use.
        model_name (str): The name of the pretrained model to use.
        train_opt (bool): Whether or not to train the model (default True).
        eval_opt (bool): Whether or not to evaluate the model (default True).
        save (bool): Whether or not to save the model and history (default True).
        overwrite (bool): Whether or not to overwrite existing files (default True).
        batch_size (int): The batch size to use (default 16).
        learning_rate (float): The learning rate to use (default 0.0001).
        epochs (int): The number of epochs to train for (default 100).
        printing (bool): Whether or not to print progress (default True).
        return_acc (bool): Whether or not to return the accuracy (default False).
        checkpoint_path (str): The path to save the model checkpoint (default './results/checkpoints').
        log_path (str): The path to save the training history (default './results/log').

    Returns:
        The maximum validation accuracy achieved during training, or 0 if training is skipped.
    """
    # try to use gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if printing:
        print('*'*50)
        print(' '*50, 'model name:    ', model_name.upper())
        print(' '*50, 'dataset name:  ', dataset_name.upper())
        print(' '*50, 'batch size:    ', batch_size)
        print(' '*50, 'learning rate: ', learning_rate)
        print(' '*50, 'epochs:        ', epochs)
        print(' '*50, 'device:        ', device)
        print('*'*50)
    # save paths
    model_save_path = f'{checkpoint_path}/{exp}-{dataset_name}-{model_name}-best-model.pt'
    history_save_path = f'{log_path}/{exp}-{dataset_name}-{model_name}-history.csv'
    eval_save_path = f'{evals_path}/{exp}-{dataset_name}-{model_name}-eval.txt'
    
    # exit the function if the overwrite not allowed
    if not overwrite and save and os.path.exists(history_save_path):
        if printing:
            print('[INFO]  The history file of this experiment is already exist.')
        if return_acc:
            return max(pd.read_csv(history_save_path)['val_accuracy'])
        else:
            return 0
    # create folders if does not exists
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(evals_path):
        os.makedirs(evals_path)
    # train/eval options
    if train_opt or eval_opt:
        # get train and val dataloaders
        dataset = ImageDataset(dataset_name, batch_size, image_size, printing=printing)
        # initialize the model
        model = pretrained_network(model_name, len(dataset.classes))
        model.to(device)
        # loss function and optimizer
        criterion = CrossEntropyLoss()
        optimizer_ft = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)        
        if train_opt:
            # training
            if printing:
                print('\n\n[INFO]  Model Training...')
            
            model, history = train(model, criterion, optimizer_ft, scheduler, dataset, device, epochs, printing, comet)
            if save:
                # save model and training history
                torch.save(model.state_dict(), model_save_path)
                pd.DataFrame.from_dict(history).to_csv(history_save_path, index=False)
                if printing:
                    print('\n[INFO]  Saved:  ', model_save_path)
                    print('[INFO]  Saved:  ', history_save_path)
        if eval_opt:
            # evaluation
            if printing:
                print('\n\n[INFO]  Model Evaluation...')
            accuracy, precision, recall, f1, loss = evaluate(model, criterion, dataset, device, printing=printing)
            
            txt = "\n\n[INFO]  Accuracy:  {:.4f}\n[INFO]  Precision: {:.4f}\n[INFO]  Recall:    {:.4f}\n[INFO]  F1 Score:  {:.4f}\n[INFO]  Loss:      {:.4f}\n".format(accuracy, precision, recall, f1, loss)
            with open(eval_save_path, "+w") as f:
                f.write(txt)
            if printing:
                print(txt)

        if return_acc:
            return max(history['val_accuracy'])

        