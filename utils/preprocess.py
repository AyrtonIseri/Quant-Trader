import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data_split import splitter
from collections import deque
from tqdm import tqdm
from random import shuffle

def _assert_df_format(dataframe: pd.DataFrame):
    if 'Classification' not in dataframe.columns:
        raise KeyError('Classification column not found in offered dataframe')

class RNNDataset(Dataset):
    def __init__(self, features: torch.tensor, labels: torch.tensor, seq_len: int, transform = None, 
                 target_transform = None, beg: int = 0, end: int = None, train = False):

        if end is None:
            end = len(labels) - seq_len

        if ((end - beg) > len(labels)) or (beg < 0) or (end > len(labels)):
            raise KeyError(f'tensor indexing is currently unnaceptable at beginning: {beg} and ending: {end} with tensor length of {len(labels)}')
        
        self._labels = labels
        self._features = features
        self._seq_len = seq_len
        self._transform = transform
        self._target_transform = target_transform
        self._beg = beg
        self._end = end
        self._train = train

    def __len__(self):
        return self._end - self._beg + 1

    def __getitem__(self, idx: int):
        
        sequence = self._features[idx+self._beg: idx+self._seq_len+self._beg]
        
        label = self._labels[idx+self._seq_len - 1 + self._beg]

        no_of_classes = self._labels.max().item() + 1

        if self._transform:
            sequence = self._transform(sequence)
        
        if self._target_transform:
            label = self._target_transform(label)
        
        if self._train:
            label = one_hot(label, num_classes=no_of_classes)

        return sequence, label


class preprocess:
    '''
    preprocessor class. Its job is to scale, store and sequentiate data properly.
    
    Methods applied:
    #1 Date is handled
    #2 dataframe is splitted in test and train
    #3 features are scaled (no data leakage)
    #4 datasets are sequentiated
    
    one hot encoding, data balancing, and shuffling should be performed during processing step.
    '''

    def __init__(self, sequence_length: int):
        '''
        :args:

        sequence_length: length of the sequence to be created in the RNN model.
        '''

        self._seq_len = sequence_length

    def preprocess(self, dataframe: pd.DataFrame, validation_size: float = 0.10, testing_size : float = 0.10) -> tuple[RNNDataset, RNNDataset]:
        '''
        preprocessing method. Will first scale, then normalize and finally sequentiate the data. Returns X_train, y_train, X_test, y_test

        :args:

        dataframe: pandas dataframe containing a 'Classification' column.
        '''

        _assert_df_format(dataframe)
        dataframe.reset_index(inplace=True, drop=True)
        dataframe = self._handle_date(dataframe)

        #Inputs and labels of the classification
        features = dataframe.drop('Classification', axis = 1)
        labels = dataframe.Classification
        
        features_average = np.array([average for average in features.mean()])
        features_std = np.array([std for std in features.std()])
        features = self._normalize(features, mean=features_average, std=features_std)

        features = torch.from_numpy(features.to_numpy()).float()
        labels = torch.from_numpy(labels.to_numpy()).to(torch.int64)

        #getting datasets' indexes on tensors.
        val_ds_beg, test_ds_beg = self._get_splitters(dataframe, validation_size, testing_size) #validation and test dataset tensor beginning index

        train_ds_end = val_ds_beg - 1
        val_ds_end = test_ds_beg - 1

        #Creating train, validation and test custom datasets
        train_dataset = RNNDataset(features=features, labels=labels, seq_len=self._seq_len, 
                                   end=train_ds_end, train=True)
        
        validation_dataset = RNNDataset(features=features, labels=labels, seq_len=self._seq_len,
                                        beg=val_ds_beg, end=val_ds_end)

        test_dataset = RNNDataset(features=features, labels=labels, seq_len=self._seq_len,
                                  beg=test_ds_beg)

        return train_dataset, validation_dataset, test_dataset
        
    def _get_splitters(self, dataframe, validation_size, test_size):
        len_ = len(dataframe) - self._seq_len + 1

        vs_beg = round(len_ * (1 - validation_size - test_size)) #new_tensor
        ts_beg = round(len_ * (1 - test_size)) #new_tensor

        return vs_beg, ts_beg

    def _handle_date(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        method to handle dates in the sequenced data (temporarilly deleting this axis)
        '''
        if 'time' in dataframe.columns:
            return dataframe.drop('time', axis=1)
        return dataframe

    def _normalize(self, X: pd.DataFrame, mean: np.array, std: np.array):
        '''
        Normalizes the sequences dataset.

        :args:

        X: dataframe to be normalized. Size: (instances, features)
        mean: array containing the mean. Size: (features)
        std: array containing the standard deviation. Size: (features)
        '''

        return (X - mean)/std


