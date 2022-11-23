from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


PATH = './name_gender_dataset.csv'
DATASET = pd.read_csv(PATH)


SRC = DATASET.Name.values.tolist()
TRG = DATASET.Gender.values.tolist()


TRAIN_DATA, TEST_DATA, TRAIN_LABEL, TEST_LABEL = train_test_split(
    SRC, TRG, test_size=0.2, random_state=42, shuffle=True)
TRAIN_DATA, VALID_DATA, TRAIN_LABEL, VALID_LABEL = train_test_split(
    TRAIN_DATA, TRAIN_LABEL, test_size=0.1, random_state=42, shuffle=True)


char_set = set()

for name in TRAIN_DATA:
    for ch in list(name):
        char_set.add(ch.lower())


def indexing(name, mapping_data):
    return [mapping_data[char.lower()] if char.lower() in mapping_data else mapping_data['<UNK>'] for char in name]


def pad_collate(batch):
    (data, label) = zip(*batch)
    data = [torch.tensor(i, dtype=torch.int) for i in data]
    label = torch.tensor(label, dtype=torch.float)

    data = pad_sequence(data, batch_first=True, padding_value=0)
    data = torch.tensor(data, dtype=torch.int)

    return data, label

# print(char_set)


CHAR2INDEX = {char: indx + 2 for indx, char in enumerate(char_set)}
CHAR2INDEX['<UNK>'] = 1  # for OOV
CHAR2INDEX['<PAD>'] = 0  # for PADDING


INDEX2CHAR = {indx + 2: char for indx, char in enumerate(char_set)}
INDEX2CHAR[1] = '<UNK>'  # for OOV
INDEX2CHAR[0] = '<PAD>'  # for PADDING

# print(CHAR2INDEX)
# print(INDEX2CHAR)

LABEL2INDEX = {'M': 0, 'F': 1}
INDEX2LABEL = {0: 'M', 1: 'F'}

# print(LABEL2INDEX)
# print(INDEX2LABEL)

# set device to CUDA if available, else to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Device:', DEVICE)

"""# **Create a Dataset class**"""


class DATASET(Dataset):
    def __init__(self, names, genders, mapping_data, mapping_label):
        self.names = names
        self.names = [indexing(name, mapping_data) for name in names]
        self.genders = genders
        self.genders = [mapping_label[gender] for gender in self.genders]
        self.length = len(self.names)

    def __getitem__(self, item):
        return self.names[item], self.genders[item]

    def __len__(self):
        return self.length


DATASET_TRAIN = DATASET(names=TRAIN_DATA, genders=TRAIN_LABEL, mapping_data=CHAR2INDEX,
                        mapping_label=LABEL2INDEX)
DATASET_VALID = DATASET(names=VALID_DATA, genders=VALID_LABEL, mapping_data=CHAR2INDEX,
                        mapping_label=LABEL2INDEX)
DATASET_TEST = DATASET(names=TEST_DATA, genders=TEST_LABEL, mapping_data=CHAR2INDEX,
                       mapping_label=LABEL2INDEX)


# (DATASET_TRAIN.names)

# print(DATASET_TRAIN.genders)


BATCH_SIZE = 32

'''
Data Loader
'''
DATALOADER_TRAIN = DataLoader(dataset=DATASET_TRAIN,
                              batch_size=BATCH_SIZE,
                              num_workers=4,
                              collate_fn=pad_collate)

DATALOADER_TEST = DataLoader(dataset=DATASET_TEST,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             collate_fn=pad_collate)

DATALOADER_VALID = DataLoader(dataset=DATASET_VALID,
                              batch_size=BATCH_SIZE,
                              num_workers=4,
                              collate_fn=pad_collate)

# iter(DATALOADER_TRAIN).next()
