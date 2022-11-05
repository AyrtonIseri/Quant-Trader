import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.data_split import splitter
from collections import deque
from tqdm import tqdm
from random import shuffle
pd.set_option('use_inf_as_na',True)

def _assert_df_format(dataframe: pd.DataFrame):
    if 'Classification' not in dataframe.columns:
        raise KeyError('Classification column not found in offered dataframe')

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

    def preprocess(self, dataframe: pd.DataFrame, validation_size: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        preprocessing method. Will first scale, then normalize and finally sequentiate the data. Returns X_train, y_train, X_test, y_test

        :args:

        dataframe: pandas dataframe containing a 'Classification' column.
        '''

        _assert_df_format(dataframe)
        dataframe.reset_index(inplace=True, drop=True)
        dataframe = self._handle_date(dataframe)

        dataframe_splitter = splitter(validation_size=validation_size)
        X_train, y_train, X_test, y_test = dataframe_splitter.split_dataset(dataframe=dataframe, include_targets=True)
        
        X_train, X_test = self._scale(X_train, X_test)

        training_dataset = self._sequentiate(X_train, y_train, sequence_length= self._seq_len)
        testing_dataset = self._sequentiate(X_test, y_test, sequence_length= self._seq_len)

        return training_dataset, testing_dataset
        

    def _handle_date(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        method to handle dates in the sequenced data (temporarilly deleting this axis)
        '''
        if 'Day' in dataframe.columns:
            return dataframe.drop('Day', axis=1)


    def _scale(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Method to standard-scale (0 mean - 1 std) the data according to the X_train dataframe. 
        
        This method assures that data leakage won't happen

        :args:

        X_train: dataframe that contains the train data
        
        X_test: dataframe that contains the test data
        '''
        for column in X_train.columns:
            scaler = StandardScaler()
            scaler.fit(X_train[column].to_numpy().reshape(-1, 1))
            X_train[column] = scaler.transform(X_train[column].to_numpy().reshape(-1, 1))
            X_test[column] = scaler.transform(X_test[column].to_numpy().reshape(-1, 1))
        
        return X_train, X_test

    def _normalize(self, series: pd.Series) -> pd.Series:
        '''
        normalizes the data by using pandas' pct_change method.

        :args:

        series: pandas series containing the data to be normalized
        '''
    
        new_series = series.pct_change()
        
        return new_series

    def _sequentiate(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> np.array:
        '''
        Sequentiate the data according in a double-ended-queue(deque). Returns a numpy array containing (sequence, target)

        :args:
        
        X_df: dataframe to make the sequence up

        y_df: dataframe, series or numpy containing the targets in the same indexing style as X

        sequence_length: size of the sequence in the training model.

        '''

        stack = deque(maxlen = sequence_length)
        sequential_data = []

        for idx, values in enumerate(X.values, start=X.index.min()):
            stack.append(values)
            if len(stack) == sequence_length:
                sequential_data.append([np.array(stack), y[idx]])

        return np.array(sequential_data)
