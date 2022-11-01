import pandas as pd
from models.Model import Model

class Agent:
    '''
    Abstraction of the Agent class. Agents and other abstractions should inherit from this class
    '''
    def __init__(self, account_balance: float, models_weights: list[float], models_kwargs: list[dict],
                 historical_data: pd.DataFrame):
        '''
        Asserts that arguments are consistent and initiate all models listed.

        :args:

        account_balance: Initial investments hold by the agent
        models_weights: list of weights per model to distribute funds. Assure that sum = 1
        models_kwargs: list of dictionary of keywords to initiate the models. Make sure that all parameters
        are the same as stated in Model.py
        historical_data: data to run the agent and all models.
        '''
        
        assert(len(models_weights) == len(models_kwargs))
        assert(sum(models_weights) == 1)

        self._no_of_models = len(models_weights)
        self._account_balance = account_balance
        self._data = historical_data

        self._models = [Model(**model_kwargs, historical_data=historical_data, 
                        invested_value = account_balance * model_weight) for (model_kwargs, model_weight) in 
                        zip(models_kwargs, models_weights)]



    def run(self):
        '''
        execute all models in stored by the agent
        '''

        for model in self._models:
            model.run_model()


    def get_no_of_models(self) -> int:
        '''
        returns the current number of model hold by the agent
        '''
        
        return self._no_of_models

    def get_balance(self) -> float:
        '''
        return the final balance of the agent
        '''

        balance = 0
        for model in self._models:
            balance += model.get_account_balance()

        return balance


    def get_daily_results(self) -> list[pd.DataFrame]:
        '''
        return a list of dataframes of size no_of_models that contain all results per model per day (in a dataframe format).
        '''
        
        results = [model.get_daily_results() for model in self._models]
        
        return results