import numpy as np
import pandas as pd

class splitter:
    '''
    class created to split a sequential dataframe (time series more specifically), without shuffling features
    '''

    def __init__(self, validation_size:float, testing_size: float = 0):
        '''
        init method for splitter class.

        :args:

        validation_size: Percentual size (in relation to dataframe) of the testing dataset.
        '''
        if (validation_size > 1) or (validation_size < 0) or (testing_size+validation_size > 1):
            raise ValueError(f'Validation size should be a percentual value between 0 and 100%. Assign value is at {validation_size}.')

        self._validation_size = validation_size
        self._testing_size = testing_size

    def split_dataset(self, dataframe: pd.DataFrame, include_targets: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        split method to separate train and test datasets. Returns a tuple (train_dataset, test_dataset)

        :args:

        dataframe: dataframe that requires splitting method.
        
        include_targets: bool variable that indicates whether x and y datasets should be segregated.
        '''

        train_dataset_size = int(len(dataframe) * (1 - self._validation_size - self._testing_size))
        validation_dataset_size = len(dataframe) * self._validation_size
        test_dataset_size = len(dataframe) * self._testing_size

        train_dataset = dataframe.iloc[:train_dataset_size]
        validation_dataset = dataframe.iloc[train_dataset_size:-test_dataset_size]
        
        if self._testing_size:
            test_dataset = dataframe.iloc[-test_dataset_size:]
        else:
            test_dataset = None    

        if include_targets:
            x_train_dataset, y_train_dataset = train_dataset.drop('Classification', axis=1), train_dataset.Classification
            x_validation_dataset, y_validation_dataset = validation_dataset.drop('Classification', axis=1), validation_dataset.Classification

            if self._testing_size:
                x_test_dataset, y_test_dataset = test_dataset.drop('Classification', axis=1), test_dataset.Classification
                return (x_train_dataset, y_train_dataset, x_validation_dataset, y_validation_dataset, x_test_dataset, y_test_dataset)

            return (x_train_dataset, y_train_dataset, x_validation_dataset, y_validation_dataset)

        if self._testing_size:
            return (train_dataset, validation_dataset, test_dataset)

        return (train_dataset, validation_dataset)