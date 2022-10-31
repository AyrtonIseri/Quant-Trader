from abc import ABC, abstractclassmethod
import pandas as pd
from datetime import date
from utils.constants import Options
import Model

class Indicator (ABC):
    '''
    Abstraction of an indicator instance. Its main purpose is to analyze the dataset and output an operation to a model
    '''
    
    def __init__(self, model: Model):
        self._model = model

    def get_option(self, data: pd.DataFrame, current_date: date) -> Options:
        '''
        Evaluates if data is correctly formatted. If so, proceeds to calculate the daily recomendation.

        :args:

        data: Dataframe that contains daily prices of the stock

        current_date: date to perform the recommendation.
        '''

        if ('Day' not in data.columns) or ('Price' not in data.columns):
            raise Exception('Dataframe is not formatted according to convention. Consider adding "Day" and "Price" columns.')

        option = self._run_indicator(data, current_date)

        return option


    @abstractclassmethod
    def __str__(self) -> str:
        '''
        Indicator name
        '''
        ...

    @abstractclassmethod
    def _run_indicator(self, data: pd.DataFrame, current_date: date) -> Options:
        '''
        Method to implement actual indicator recommendation algorithm.
        '''
        ...
