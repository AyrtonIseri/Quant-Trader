import pandas as pd
import datetime
from utils.constants import Options
from numba import jit
import numpy as np

def load_data():

    data_path = 'historicals/BTC-Min.csv'

    data = pd.read_csv(data_path, usecols=['date', 'close', 'Volume BTC'], parse_dates=['date'], infer_datetime_format=True)
    data.rename(columns={'date': 'Day', 'close': 'Price', 'Volume BTC': 'Volume'}, inplace=True)

    return data

@jit(nopython=True)
def _made_profit(current_price: float, future_price: float, tp: float) -> bool:
    if future_price > current_price*(1+tp):
        return True
    return False

@jit(nopython=True)
def _took_loss(current_price: float, future_price: float, sl: float) -> bool:
    if future_price < current_price*(1+sl):
        return True
    return False

@jit(nopython=True)
def _position_ended(current_price: float, future_price: float, stop_loss: float, take_profit: float) -> bool:

    made_profit = _made_profit(current_price, future_price, take_profit)
    took_loss = _took_loss(current_price, future_price, stop_loss)

    return ((made_profit) or (took_loss))

@jit(nopython=True)
def get_classification_array(price_array: np.array, stop_loss: float, take_profit: float) -> np.array:
    length = len(price_array)

    new_array = np.zeros((length, 1))

    for idx in range(length):
        current_price = price_array[idx]

        second_idx = idx+1


        second_idx = idx+1

        while (second_idx < length) and (not _position_ended(current_price, price_array[second_idx], stop_loss, take_profit)):
            second_idx += 1

        if second_idx < length:
            future_price = price_array[second_idx]

            if _made_profit(current_price, future_price, take_profit):
                new_array[idx] = 1

            if _took_loss(current_price, future_price, stop_loss):
                new_array[idx] = -1

        if idx % (length//100) == 0:
            print(f'Interacao atual: {idx}/{length}')

    return new_array

class Classifier:
    def __init__(self, stop_loss: float, take_profit: float):

        assert((stop_loss < 0) and (stop_loss > -1))
        assert((take_profit >0))

        self._sl = stop_loss
        self._tp = take_profit


    def _convert_to_enum(self, integer: int) -> Options:
        if integer == 0:
            return Options.HOLD
        if integer == 1:
            return Options.BUY
        return Options.SELL

    def classify_data(self, dataframe: pd.DataFrame):
        
        dataframe['Classification'] = get_classification_array(dataframe['Price'].to_numpy(), self._sl, self._tp)
        # dataframe['Classification'] = dataframe['Classification'].apply(self._convert_to_enum)
        return dataframe



