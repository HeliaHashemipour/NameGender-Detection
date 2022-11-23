import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.functional import sigmoid
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_preprocessing import *
from model import *

import warnings

warnings.filterwarnings('ignore')
# set device to CUDA if available, else to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)


def binary_acc(y_pred, y_test):
    y_pred = torch.tensor(y_pred)
    y_test = torch.tensor(y_test)

    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = acc
    return acc
  

def save_checkpoint(save_path, model, optimizer, valid_loss):
    if not save_path:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')



def load_checkpoint(load_path, model, optimizer):
    if not load_path:
        return

    state_dict = torch.load(load_path, map_location=DEVICE)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']
  
  

""" Step 2: Construct loss and optimizer  and train the Model"""
CRITERION = torch.nn.BCELoss().to(DEVICE)
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), LEARNING_RATE)
PATH_MODEL = '/content/drive/MyDrive/best_model.pt'

plot_loss_train =[]
plot_loss_valid =[]

best_accuracy = 0.0
CHANGE_VALID_ACCURACY = 0

for epoch in tqdm(range(EPOCH_SIZE)):
    print(f"#Epoch {epoch} is running ...")
    MODEL.train()
    loss_train_BATCH = []

    for BATCH_INDEX, BATCH in enumerate(DATALOADER_TRAIN):
        DATA_BATCH, LABEL_BATCH = BATCH
        # print(type(DATA_BATCH))
        DATA_BATCH = DATA_BATCH.to(DEVICE)
        LABEL_BATCH = LABEL_BATCH.to(DEVICE)
        out = MODEL(DATA_BATCH)

        LOSS = CRITERION(out, LABEL_BATCH)
        loss_train_BATCH.append(LOSS.cpu().detach().numpy())
        
        LOSS.backward()
        OPTIMIZER.step()

        OPTIMIZER.zero_grad()
    plot_loss_train.append(sum(loss_train_BATCH) / len(loss_train_BATCH))
        
    MODEL.eval()

    with torch.no_grad():
            loss_valid_BATCH = []

            valid_TRUE_LABLES = []
            valid_predicted_LABLES = []

            for BATCH in DATALOADER_VALID:
                DATA_BATCH, LABEL_BATCH = BATCH
                DATA_BATCH = DATA_BATCH.to(DEVICE)
                LABEL_BATCH = LABEL_BATCH.to(DEVICE)

                OUTPUT = MODEL(DATA_BATCH)
                LOSS = CRITERION(OUTPUT, LABEL_BATCH)
                loss_valid_BATCH.append(LOSS.cpu().detach().numpy())

                valid_TRUE_LABLES.extend(LABEL_BATCH.tolist())
                valid_predicted_LABLES.extend(torch.round(OUTPUT).tolist())

            ACCURACY = binary_acc(y_pred=valid_predicted_LABLES, y_test=valid_TRUE_LABLES)
            print(f'Accuracy on Validation Set: {ACCURACY :.2f}%')

            if ACCURACY > best_accuracy:
                best_accuracy = ACCURACY
                save_checkpoint(save_path=PATH_MODEL, model=MODEL,
                                optimizer=OPTIMIZER, valid_loss=plot_loss_valid)
            elif ACCURACY == best_accuracy:
                CHANGE_VALID_ACCURACY += 1

            if CHANGE_VALID_ACCURACY == 3:
                print(f'Change Valid Accuracy: True --> Early Stopped.')
                break

            plot_loss_valid.append(sum(loss_valid_BATCH) / len(loss_valid_BATCH))

    print(f'The best validation accuracy is: {best_accuracy :.2f}%')
    print(f"Training is done.")

"""# **Analyze the results**"""

from matplotlib import pyplot as plt

plt.title("LOSS Curve")

plt.plot(plot_loss_train, label='Train Loss')
plt.plot(plot_loss_valid, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
plt.savefig('/content/drive/MyDrive/gender_detection_loss_curve.png')

