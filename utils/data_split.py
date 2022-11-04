import numpy as np
import pandas as pd

class splitter:
    '''
    class created to split a sequential dataframe (time series more specifically), without shuffling features
    '''

    def __init__(self, validation_size):
        '''
        init method for splitter class.

        :args:

        validation_size: Percentual size (in relation to dataframe) of the testing dataset.
        '''
        if (validation_size > 1) or (validation_size < 0):
            raise ValueError(f'Validation size should be a percentual value between 0 and 100%. Assign value is at {validation_size}.')

        self._validation_size = validation_size

    def split_dataset(self, dataframe: pd.DataFrame, include_targets: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        split method to separate train and test datasets. Returns a tuple (train_dataset, test_dataset)

        :args:

        dataframe: dataframe that requires splitting method.
        include_targets: bool variable that indicates whether x and y datasets should be segregated.
        '''
        
        sorted_df = dataframe.sort_values('date')

        train_dataset_size = int(len(dataframe) * (1 - self._validation_size))
        test_dataset_size = len(dataframe) - train_dataset_size


        train_dataset = sorted_df.iloc[:train_dataset_size]
        test_dataset = sorted_df.iloc[-test_dataset_size:]

        if include_targets:
            x_train_dataset, y_train_dataset = train_dataset.drop('Classification', axis=1), train_dataset.Classification
            x_test_dataset, y_test_dataset = test_dataset.drop('Classification', axis=1), test_dataset.Classification

            return (x_train_dataset, y_train_dataset, x_test_dataset, y_test_dataset)

        return (train_dataset, test_dataset)