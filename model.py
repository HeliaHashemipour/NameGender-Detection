from torch.nn.functional import sigmoid
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

import warnings

warnings.filterwarnings('ignore')
# set device to CUDA if available, else to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

""" Hyper-Parameter Definition """

BATCH_SIZE = 50
EPOCH_SIZE = 30

INPUT_SIZE = len(CHAR2INDEX)
HIDDEN_SIZE = 256
OUTPUT_DIM = 1
N_LAYERS = 1
LEARNING_RATE = 0.001


class MODEL_Gender(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, n_layers, batch_size):
        super(MODEL_Gender, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, self.hidden_size)

        self.lstm = nn.LSTM(self.hidden_size,
                            self.hidden_size,
                            self.n_layers
                            )

        self.fully_connected = nn.Linear(hidden_size,
                                         output_dim)

    def forward(self, input):
        # input: [batch size, sent len]  --> [sent len, batch size]
        # print(input.size(0))

        # print(input.T)
        self.sent_len = input.size(0)
        input = input.T

        # Embedding: [sent len, batch size] --> [sent len, batch size, hidden size]
        embedded = self.embedding(input)
        self.batch_size = embedded.size(1)
        # output = [sent len, batch size, hidden size]
        # hidden =  [Direction * num_layers, Batch size, hidden size]
        output, (hidden, cell) = self.lstm(embedded)

        fc_output = self.fully_connected(hidden[-1, :])
        fc_output = fc_output.view(-1).to(DEVICE)

        return sigmoid(fc_output)


""" Step 1: Model class in PyTorch way """
MODEL = MODEL_Gender(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    batch_size=BATCH_SIZE
)
# MODEL.to(DEVICE)

# MODEL.forward
