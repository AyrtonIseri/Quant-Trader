from datetime import date
from Indicators.Indicator import Indicator
from utils.position import Position
import pandas as pd
import numpy as np

class AbstractModel:
    '''
    Model skeleton that should abstract all models interactions with signals
    '''

    def __init__(self, invested_value: float, 
                 take_profit: float, stop_loss: float, 
                 indicator: Indicator, historical_data: pd.DataFrame, 
                 invest_percentage: float = 0.2):
        '''
        Instantiate a model with its parameters.

        :args:

        invested_value: value invested in the wallet
        take_profit: margin to realize profits
        stop_loss: negative margin to hault the position
        indicator: indicator object necessary to recommend buy/sell positions
        historical_data: pandas dataframe with 'Day' and 'Price' columns to perform the trading
        invest_percentage: percentage of wallet money to invest in each new position
        '''

        self._positions = []
        self._account_balance = np.array([invested_value])
        self._take_profit = take_profit
        self._stop_loss = stop_loss
        self._percentage_to_invest = invest_percentage
        self._no_of_positions = 0
        self._indicator = indicator
        self._data = historical_data

    def create_position(self, price: float, value_to_invest:float, date: date):
        '''
        Creates a new position to bet in the current model

        :args:

        price: Price in which the new position was made.
        value_to_invest: amount of money betted in the position
        date: day of the position's creation
        '''

        new_position = Position(entry_price=price, invested_value=value_to_invest, initial_date=date)
        self._positions.append(new_position)
        self._no_of_positions += 1

    def _end_position(self, position_idx: int, date: date):
        self._positions[position_idx].close_position(final_date = date)



    