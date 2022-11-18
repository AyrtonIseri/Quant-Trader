import pandas as pd
import numpy as np
from numba import jit


def load_BTC():

    data_path = 'historicals/btcusd.csv'

    # data = pd.read_csv(data_path, usecols=['date', 'close', 'open', 'high', 'low', 'Volume USD', 'Volume BTC'], parse_dates=['date'], infer_datetime_format=True)
    data = pd.read_csv(data_path, usecols=['time', 'close', 'volume'])
    data.rename(columns={'close': 'Price_BTC', 'volume': 'Volume_BTC'}, inplace=True)

    data.sort_values('time', inplace=True)

    return data

def load_ETH():
    data_path = 'historicals/ethusd.csv'

    data = pd.read_csv(data_path, usecols=['time', 'close', 'volume'])
    data.rename(columns={'close': 'Price', 'volume': 'Volume_ETH'}, inplace=True)

    data.sort_values('time', inplace=True)

    return data

def load_data():
    BTC = load_BTC()
    ETH = load_ETH()

    data = pd.merge(left = ETH, right = BTC, how = 'left', on = 'time').dropna()
    data['time'] = pd.to_datetime(data.time, unit='ms')

    return data

    
def shift(arr, num) -> np.array:
    '''
    shifts a numpy array and replaces empty values with NaNs

    args:

    arr: array to be shifted
    num: number of shifts to perform
    '''

    fill_value=np.nan
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr

    result = result[~np.isnan(result)]

    return result

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

def get_strict_class(price_data: np.array, time_window: int, extra_class: bool = False, threshold: float = None) -> np.array:
    '''
    function to operate inside an array and classify based on a future time window whether the stock raised or dropped.

    args:

    price_data: numpy array containing the historical data.

    time_window: time value to perform the classification

    extra_class: boolean to consider or not more than 2 classes.

    threshold: min vaue to be classified as a new class.
    '''

    if extra_class:
        classify = lambda n: 0 if n < -threshold else 1 if n < 0 else 2 if n < threshold else 3
    else:
        classify = lambda n: 1 if n >= 0 else 0

    classify = np.vectorize(classify)

    future_prices = shift(price_data, -time_window)

    price_data = price_data[:-time_window]

    diff_array = (future_prices - price_data)/price_data
    return classify(diff_array)


class Classifier:
    def __init__(self, dataframe: pd.DataFrame):
        '''
        Classifier object. Receives a dataframe containing the price variable and classifies whether
        an event occured. The classification is returned in a new column called 'Classification'
        This event can be a raise, a drop, a maintainance in any personalized manner.

        args:

        dataframe: pd.Dataframe containing a 'price' column to be classified.
        '''
        
        assert('Price' in dataframe.columns)

        self._dataframe = dataframe

    def classify_data_rd(self, max_drop: float, max_raise: float, max_period: int = 10):
        '''
        Classifies the dataset according to its maximum raises and drops in stock/share price.
        Uses the mr and md parameters to determine when an event occured during a max_period.
        '''

        assert((max_drop < 0) and (max_drop > -1))
        assert((max_raise >0))
        assert((max_period>0))

        res = self._dataframe
        res['Classification'] = get_classification_array(self._dataframe['Price'].to_numpy(), max_drop, max_raise, max_period)

        return res

    def classify_data_strict_time(self, time_outlook: int, extra_classes: bool = False, threshold: float = None):
        '''
        Classify on a strict future basis whether a stock raised or dropped. If desired, there can still be another classification for
        rocket-raises and stock-plummets

        decoding:
        
        in case there is no extra class:
        0 -> dropped
        1 -> raised

        otherwise:
        0 -> big drop
        1 -> common drop
        2 -> common raise
        3 -> big raise

        args:

        time_outlook: period of time to analyze the stock movement. Time units will be the same as dataframe's date column.

        extra_clases: indicates if the model should classify big stock movements. Defaults to false.
        If set to true, threshold parameter should be passed as well.

        threshold: percentual float that determines what is considered a movement big enough to move to another class. 
        Must be passed if extra_class is true. 
        '''

        res = self._dataframe
        

        if extra_classes:
            if threshold is None:
                raise ValueError('if extra classification is wanted, you must pass a threshold variable')

            if type(threshold) != float:
                raise TypeError(r'Threshold should be a float specifying the min. stock movement required (in % change) ')

            assert(threshold > 0)
        
            Classification = get_strict_class(res['Price'].to_numpy(), time_window=time_outlook,
                                                     extra_class=True, threshold=threshold)
        
        else:
            Classification = get_strict_class(res['Price'].to_numpy(), time_window=time_outlook)

        res = res[:-time_outlook]
        res.loc[:, 'Classification'] = Classification

        return res
