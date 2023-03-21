#  +-+ +-+ +-+ +-+ +-+ +-+
#  |i| |m| |p| |o| |r| |t|
#  +-+ +-+ +-+ +-+ +-+ +-+

import torch

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import os

from networks import pretrained_network

#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+
#  |v| |i| |s| |u| |a| |l| |i| |z| |a| |t| |i| |o| |n|
#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+


def dist_plot(dataset_name, model_name, trained_epochs, filter_epoch, save_path='./results/vis', log_path='./results/log'):
    """
    Plots the validation accuracy of a given model trained on a specific dataset over a certain number of epochs.

    Args:
        dataset_name (str): The name of the dataset the model was trained on.
        model_name (str): The name of the model.
        trained_epochs (int): The total number of epochs the model was trained for.
        filter_epoch (int): The number of epochs to plot up to.
        save_path (str): The path to save the resulting plot image. Default is './results/vis'.
        log_path (str): The path to the directory containing the history CSV files. Default is './results/log'.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the log_path directory does not exist.

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # get history files names    
    csv_files = [csv_file for csv_file in os.listdir(log_path) if f'{dataset_name}-{model_name}' in csv_file and str(trained_epochs) in csv_file]
    # load history files
    dfs = [pd.read_csv(os.path.join(log_path, csv_file)) for csv_file in csv_files]    
    # plot dist lines
    plt.figure(figsize=(12, 5))
    for i, df in enumerate(dfs):
        # filter epochs
        df = df.head(filter_epoch)
        # get init name and labels
        best_acc = round(max(df['val_accuracy']), 3)
        label = '{}'.format(best_acc)
        # add current df val_accuracy progress to the figure
        plt.plot(df['epoch'], df['val_accuracy'], label=label)
    plt.title(' '.join([dataset_name, model_name, str(filter_epoch), 'epochs']))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f'{save_path}/dist-{dataset_name}-{model_name}-epochs-{filter_epoch}-of-{trained_epochs}.png', dpi=360, bbox_inches='tight')
    plt.show()


def progress_plot(dataset_name, model_name, epochs=None, save_path='./results/vis', log_path='./results/log'):
    """
    Plots the training and validation accuracy over epochs for a given model trained on a specific dataset.

    Args:
        dataset_name (str): The name of the dataset the model was trained on.
        model_name (str): The name of the model.
        epochs (int, optional): The number of epochs to plot. If not provided, the function will use the CSV file with the highest number of epochs in the log_path directory.
        save_path (str): The path to save the resulting plot image. Default is './results/vis'.
        log_path (str): The path to the directory containing the history CSV files. Default is './results/log'.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the log_path directory does not exist.

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not epochs:
        csv_files = [csv_file for csv_file in os.listdir(log_path) if f'{dataset_name}-{model_name}' in csv_file]
        epochs = max(int(name.split('-')[5]) for name in csv_files)
    csv_path = f'{log_path}/{dataset_name}-{model_name}-epochs-{epochs}-history.csv'
    df = pd.read_csv(csv_path)
    # plot lines
    plt.figure(figsize=(12, 5))
    plt.plot(df['epoch'], df['accuracy'], label='train: ' + str(round(max(df['accuracy']), 4)))
    plt.plot(df['epoch'], df['val_accuracy'], label='val:     ' + str(round(max(df['val_accuracy']), 4)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{save_path}/progress-dist-{dataset_name}-{model_name}-epochs-{epochs}.png', dpi=360, bbox_inches='tight')
    plt.show()


def cm_plot(dataset, model_name, epochs, save_path='./results/vis'):
    '''
    Plot confusion matrix of the validation set for a given dataset and trained model.

    Args:
        dataset (Dataset): validation dataset.
        model_name (str): name of the trained model.
        epochs (int): number of epochs to be loaded.
        save_path (str, optional): path to save the resulting image. Defaults to './results/vis'.

    Returns:
        None
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load model
    model_path = f'./results/checkpoints/{dataset.name}-{model_name}-epochs-{epochs}-best-model.pt'
    model = pretrained_network(model_name, len(dataset.classes))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # prediction
    y_pred = []
    y_true = []
    for inputs, labels in dataset.dataloaders['val']:
        output = model(inputs) # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # normalize
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # to dataframe
    df_cm = pd.DataFrame(cmn/np.sum(cmn) *10, index = [i for i in dataset.classes], columns = [i for i in dataset.classes])
    
    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=dataset.classes, yticklabels=dataset.classes, cmap='Blues')
    plt.title(' '.join([dataset.name, model_name, str(epochs), 'epochs']))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    plt.savefig(f'{save_path}/cm-dist-{dataset.name}-{model_name}-{epochs}-epochs.png', dpi=360, bbox_inches='tight')

