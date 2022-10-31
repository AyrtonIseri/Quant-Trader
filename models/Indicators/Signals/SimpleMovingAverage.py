from Signal import Signal
from models.Model import Model
from utils.constants import Options
import numpy as np

class SimpleMovingAverage (Signal):
    '''
    Indicator that recommends trade based on Simple Moving Average Crossing
    '''
    def __init__(self, fast_curve: int, slow_curve: int):
        '''
        :args:

        fast_curve: days to consider in rolling_mean of fast curve
        slow_curve: days to consider in rolling_mean of slow curve

        '''
        super().__init__()
        self._fc = fast_curve
        self._sc = slow_curve

    def __str__(self) -> str:
        '''
        Indicator model naming
        '''

        return f'SMVC: {self._fc} as fast and {self._sc} as slow \n'

    def _run_indicator(self, data, current_date) -> Options:
        '''
        Algorithm to perform the suggestion of BUY/SELL/HOLD

        :args:

        data: Pandas dataframe containing the stock/coin price per day.
        current_date: Date to evaluate if should buy, sell or hold.

        '''

        data = data.loc[data.Day <= current_date]

        fast_curve = data['Price'].rolling(window=self._fc).mean()
        slow_curve = data['Price'].rolling(window=self._sc).mean()

        diff_curve = (fast_curve - slow_curve)

        lastest_diff = np.sign(diff_curve[-2])
        current_diff = np.sign(diff_curve[-1])

        if current_diff > lastest_diff:
            return Options.BUY

        if current_diff < lastest_diff:
            return Options.SELL

        return Options.HOLD
