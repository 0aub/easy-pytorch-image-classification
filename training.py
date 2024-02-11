#  +-+ +-+ +-+ +-+ +-+ +-+
#  |i| |m| |p| |o| |r| |t|
#  +-+ +-+ +-+ +-+ +-+ +-+

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import copy
import os

from utils import cm_to_dict, performance_report
from datasets import ImageDataset
from networks import pretrained_network

# for "UserWarning: Truncated File Read"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+
#  |t| |r| |a| |i| |n| |i| |n| |g|
#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+


def train(model, criterion, optimizer, scheduler, dataset, device, num_epochs=10, printing=True, comet=None):
    '''
    Train the given model on a given dataset with the specified criterion, optimizer, and scheduler.

    Args:
        model: A PyTorch model to train.
        criterion: A loss function to optimize the model.
        optimizer: An optimizer to use for updating the model weights.
        scheduler: A scheduler to adjust the learning rate during training.
        dataset: A PyTorch dataset object containing the training and validation data.
        device: A PyTorch device object to run the training on.
        num_epochs (optional): The number of epochs to train the model for. Default is 10.
        printing (optional): Whether to print training progress updates. Default is True.

    Returns:
        A tuple containing the trained model and a dictionary of training history, with keys: epoch, loss, accuracy, val_loss, val_accuracy.
    '''
    # history dict
    history = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'time': [],
    }
    # initial variables
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # train/val variables
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    # main training/validation loop
    for epoch in range(num_epochs):
        if printing:
            print("Epoch {}/{}".format(epoch+1, num_epochs))
        # reset epoch accuracy and loss
        epoch_timer = time.time()
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        # set model training status for the training
        model.train(True)
        # training iterations
        for i, data in enumerate(dataset.dataloaders['train']):
            print('\r\t{}/{}  time: {}s  '.format(i+1 , len(dataset.dataloaders['train']), int(time.time() - epoch_timer)), end='')
            # extract images and labels
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            # Clear gradients
            optimizer.zero_grad()
            # predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # compute loss and back propagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate training loss and accuracy
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            # free some memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        # average training loss and accuracy
        # * 2 as we only used half of the dataset
        avg_loss = loss_train / dataset.dataset_sizes['train']
        avg_acc = acc_train / dataset.dataset_sizes['train']
        # change model training status for the evaluation
        model.train(False)
        model.eval()
        cm = torch.zeros(len(dataset.classes), len(dataset.classes))
        # validation iterations
        for i, data in enumerate(dataset.dataloaders['val']):
            # extract images and labels
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            # Clear gradients
            optimizer.zero_grad()
            # predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # compute loss
            loss = criterion(outputs, labels)
            # calculate training loss and accuracy
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)
            # calculate confusion matrix
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1
            # free some memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        # average validation loss and accuracy
        avg_loss_val = loss_val / dataset.dataset_sizes['val']
        avg_acc_val = acc_val / dataset.dataset_sizes['val']
        # calculate precision, recall, and f1
        cm_dict = cm_to_dict(cm, dataset.classes)
        precision, recall, f1, _ = performance_report(cm_dict, mode = 'Macro')
        epoch_time = int(time.time() - epoch_timer)
        # printing
        if printing:
            print("loss: {:.4f}  acc: {:.4f}  val_loss: {:.4f} val_acc: {:.4f} P: {:.4f} R: {:.4f} F1: {:.4f}".format(avg_loss, avg_acc, avg_loss_val, avg_acc_val, precision, recall, f1), end='')
        # update best accuracy
        if avg_acc_val > best_acc:
            if printing:
                print("\n\tval_acc improved from {:.4f} to {:.4f}".format(best_acc, avg_acc_val))
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            if printing:
                print("\n\tval_acc did not improve from {:.4f}".format(best_acc))
        # save progress in history
        history['epoch'].append(epoch+1)
        history['loss'].append(round(np.float64(avg_loss).item(), 4))
        history['accuracy'].append(round(np.float64(avg_acc).item(), 4))
        history['val_loss'].append(round(np.float64(avg_loss_val).item(), 4))
        history['val_accuracy'].append(round(np.float64(avg_acc_val).item(), 4))
        history['precision'].append(round(np.float64(precision).item(), 4))
        history['recall'].append(round(np.float64(recall).item(), 4))
        history['f1'].append(round(np.float64(f1).item(), 4))
        history['time'].append(epoch_time)

        if comet:
            comet.log_metric('loss', round(np.float64(avg_loss).item(), 4))
            comet.log_metric('accuracy', round(np.float64(avg_acc).item(), 4))
            comet.log_metric('val_loss', round(np.float64(avg_loss_val).item(), 4))
            comet.log_metric('val_accuracy', round(np.float64(avg_acc_val).item(), 4))
            comet.log_metric('precision', round(np.float64(precision).item(), 4))
            comet.log_metric('recall', round(np.float64(recall).item(), 4))
            comet.log_metric('f1', round(np.float64(f1).item(), 4))

    # calculate training time
    if printing:
        print("\n[INFO]  Training completed in {:.0f}m {:.0f}s".format(epoch_time // 60, epoch_time % 60))
        print("[INFO]  Best accuracy: {:.4f}".format(best_acc))
    # load best weight and return it
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate(model, criterion, dataset, device, printing=True):
    """
    Evaluate the PyTorch model on the validation set of the input dataset and return evaluation metrics.

    Args:
    - model: PyTorch model to be evaluated
    - criterion: loss criterion for evaluating the model
    - dataset: input dataset containing validation set
    - device: device to use for evaluation (e.g., 'cpu', 'cuda')
    - printing: flag to print the evaluation time (default: True)

    Returns:
    - acc_test: accuracy of the model on the validation set
    - precision: precision of the model on the validation set
    - recall: recall of the model on the validation set
    - f1: f1-score of the model on the validation set
    - loss_test: testing loss of the model on the validation set
    """
    # initial variables
    since = time.time()
    loss_test = 0
    acc_test = 0
    cm = torch.zeros(len(dataset.classes), len(dataset.classes))
    # testing iterations
    for i, data in enumerate(dataset.dataloaders['val']):
        # set model training status to False for the evaluation
        model.train(False)
        model.eval()
        # extract images and labels
        inputs, labels = data
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        # predictions
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # calculate loss
        loss = criterion(outputs, labels)
        # calculate testing loss and accuracy
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)
        # calculate confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
        # free some memory
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
    # average testing loss and accuracy
    loss_test = loss_test / dataset.dataset_sizes['val']
    acc_test = acc_test / dataset.dataset_sizes['val']
    # calculate precision, recall, and f1
    cm_dict = cm_to_dict(cm, dataset.classes)
    precision, recall, f1, _ = performance_report(cm_dict, mode = 'Macro')
    if printing:
        print()
        # calculate training time 
        elapsed_time = time.time() - since
        print("[INFO]  Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    return round(acc_test.item(), 4), round(precision.item(), 4), round(recall.item(), 4), round(f1.item(), 4), round(loss_test, 4)
    
    
def evaluation_summary(exp, dataset_name, batch_size, avg_num=10, summary_save_path='./results/log/', checkpoints_path='./results/checkpoints/'):
    '''
    This function takes a dataset name, model name, the number of epochs and checkpoints path, 
    and it creates an evaluation summary CSV file for the given models in the checkpoints directory. 
    For each model, it loads the saved weights, calculates the average accuracy, precision, recall, 
    f1 score, and loss over a number of evaluations, and records the best training and validation 
    accuracy from the history. Finally, it sorts the results by the model name and saves the 
    summary as a CSV file in the specified path.

    Args:
    - dataset_name (str): The name of the dataset used in training the models.
    - avg_num (int): The number of times to evaluate the model to get the average performance.
    - summary_save_path (str): The path to save the summary CSV file.
    - checkpoints_path (str): The path to the checkpoints directory containing the saved model weights.

    Returns:
    - None: This function does not return anything. It saves the evaluation summary CSV file in the specified path.



    '''
    # get path of all models that contains the entered init, model, and epochs 
    models_paths = [checkpoints_path + name for name in os.listdir(checkpoints_path) if dataset_name in name and exp in name]
    # get dataset object
    dataset = ImageDataset(dataset_name, batch_size, printing=False)
    # define loss function
    criterion = CrossEntropyLoss()
    # try to use gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # df columns
    columns = ['filename', 'model_name', 'train_accuracy', 'val_accuracy', 'avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_loss']
    arr = []
    # loop over all saved weights
    for best_weights in tqdm(models_paths, desc='[INFO]  getting models evaluations:'):
        # record data
        filename = best_weights.split('/')[-1]
        model_name = filename.split('-')[2]
        # load the model
        model = pretrained_network(model_name, len(dataset.classes))
        model.load_state_dict(torch.load(best_weights, map_location=device))
        model.to(device)
        # compute average measures
        avg_accuracy, avg_precision, avg_recall, avg_f1, avg_loss = 0, 0, 0, 0, 0
        for i in range(avg_num):
            # get its evaluations
            accuracy, precision, recall, f1, loss = evaluate(model, criterion, dataset, device, False)
            avg_accuracy += accuracy
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            avg_loss += loss
        avg_accuracy = avg_accuracy / avg_num
        avg_precision = avg_precision / avg_num
        avg_recall = avg_recall / avg_num
        avg_f1 = avg_f1 / avg_num
        avg_loss = avg_loss / avg_num
        # read best val/train accuracy from history
        history = pd.read_csv(os.path.join(summary_save_path,f'{exp}-{dataset_name}-{model_name}-history.csv'))
        best_row = history[history['val_accuracy'] == max(history['val_accuracy'])].head(1)
        # add row to array
        arr.append([filename, model_name, round(best_row['accuracy'].item(), 4), round(best_row['val_accuracy'].item(), 4), round(avg_accuracy, 4), round(avg_precision, 4), round(avg_recall, 4), round(avg_f1, 4), round(avg_loss, 4)])
    # arr to df
    df = pd.DataFrame(arr, columns=columns)
    df = df.sort_values(by=['model_name'])
    if not os.path.exists(summary_save_path):
        os.makedirs(summary_save_path)
    summary_file_save_path = os.path.join(summary_save_path, f'{dataset_name.upper()}-evaluation_summary.csv')
    df.to_csv(summary_file_save_path, index=False)
    print('[INFO]  Saved:  ', summary_file_save_path)
