from torch import nn
import torch
from torch.utils.data import dataloader
from models.Indicators.AI import AI
from utils.preprocess import RNNDataset
import numpy as np
import pandas as pd
from utils.constants import Options
from datetime import date

class __RNN_module(nn.Module):
    def __init__(self):
        super().__init__()


    def forward():
        pass


class RNN (AI):
    '''
    Class that abstracts the Recurring Neural Network model. Should be inherited by all RNN models.
    This class will implement an LSTM model.
    '''
    def __init__(self, sequence_length: int, batch_size: int = 64, train_model: bool = True, train_dataset: RNNDataset = None,
                 n_hidden = 60):
        super().__init__()
        self._seq_len = sequence_length

        if train_model:
            if train_dataset is None:
                raise ValueError('You must provide data for the model to be trained')

        dataloader = dataloader()
        ...
        #keep this code


    def _calculate_weights(self, training_dataset: RNNDataset):
        labels = training_dataset._labels
        classes = torch.unique(labels, dim = 0)
        weights = torch.zeros((classes.shape[0],))
        total = torch.tensor(len(training_dataset))

        for idx, _class in enumerate(classes):
            weights = weights + _class * (1 - labels[labels == _class].sum() / total)

        return weights


    def _train_loop(self):
        pass

    def _predict(self):
        pass


    def __str__(self):
        '''
        Indicator name
        '''
        return f"LSTM neural network indicator"

    def _run_indicator(self, data: pd.DataFrame, current_date: date) -> Options:
        '''
        Method to implement actual indicator recommendation algorithm.
        '''

        sequence = data[data.Day <= current_date].drop('Day', inplace=True)
        sequence = sequence.iloc[-self._seq_len:].to_numpy()
        sequence = torch.from_numpy(sequence)

        recommendation = self._predict(sequence)

        if recommendation == torch.tensor([1, 0]):
            return Options.SELL

        return Options.BUY

    def min_data_size(self):
        '''
        Returns the minimal dataset size required for the algorithm to work
        '''
        return 1