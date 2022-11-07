import pandas as pd
import datetime
from utils.constants import Options
from numba import jit
import numpy as np

def load_data():

    data_path = 'historicals/BTC-Min.csv'

    data = pd.read_csv(data_path, usecols=['date', 'close', 'Volume USD'], parse_dates=['date'], infer_datetime_format=True)
    data.rename(columns={'date': 'Day', 'close': 'Price', 'Volume USD': 'Volume'}, inplace=True)

    data.sort_values('Day', inplace=True)

    return data

@jit(nopython=True)
def _made_profit(current_price: float, future_price: float, mr: float) -> bool:
    if future_price > current_price*(1+mr):
        return True
    return False

@jit(nopython=True)
def _took_loss(current_price: float, future_price: float, md: float) -> bool:
    if future_price < current_price*(1+md):
        return True
    return False

@jit(nopython=True)
def _position_ended(current_price: float, future_price: float, max_drop: float, max_raise: float) -> bool:

    made_profit = _made_profit(current_price, future_price, max_raise)
    took_loss = _took_loss(current_price, future_price, max_drop)

    return ((made_profit) or (took_loss))

@jit(nopython=True)
def get_classification_array(price_array: np.array, max_drop: float, max_raise: float, max_period: int) -> np.array:
    length = len(price_array)

    new_array = np.zeros((length, 1))

    for idx in range(length):
        current_price = price_array[idx]

        second_idx = idx+1

        while (second_idx < length) and (not _position_ended(current_price, price_array[second_idx], max_drop, max_raise)) and (second_idx - idx <= max_period):
            second_idx += 1

        if (second_idx < length) and (second_idx - idx <= max_period):
            future_price = price_array[second_idx]

            if _made_profit(current_price, future_price, max_raise):
                new_array[idx] = 1 #Price raised above max_raise

            if _took_loss(current_price, future_price, max_drop):
                new_array[idx] = 0 #Price plummeted below min_raise

        else:
            new_array[idx] = 2 #Not sufficient data or ran out of periods.

        if idx % (length//100) == 0:
            print(f'Interacao atual: {idx}/{length}')

    return new_array

class Classifier:
    def __init__(self, max_drop: float, max_raise: float, max_period: int = 10):

        assert((max_drop < 0) and (max_drop > -1))
        assert((max_raise >0))
        assert((max_period>0))

        self._md = max_drop
        self._mr = max_raise
        self._max_period = max_period

    def classify_data(self, dataframe: pd.DataFrame):

        dataframe['Classification'] = get_classification_array(dataframe['Price'].to_numpy(), self._md, self._mr, self._max_period)
        return dataframe



