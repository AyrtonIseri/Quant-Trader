from datetime import date
from utils.constants import Positions, Options
import numpy as np

class Position:
    '''
    This class represents a position that should be created and managed by a model
    '''
    def __init__(self, entry_price: float, invested_value: float, initial_date: date, option: Options):
        '''
        A position contains all information regarding the model's bet.

        :args:

        entry_price: Stock price at the moment of bet

        invested_value: amount of money invested initially

        '''

        self._option = option
        self._entry_price = entry_price
        self._initial_date = initial_date
        self._final_date = None
        self._invested_value = invested_value
        self._profit = np.array([0])
        self._state = True
    def is_open(self) -> bool:
        '''
        Return the positions current state. Either Open or Closed
        '''

        return self._state

    def close_position(self, final_date: date):
        '''
        Closes the position

        :args:
        final_date: date in which the position was closed
        '''

        self._state = False
        self._final_date = final_date


    def get_profit(self) -> float:
        '''
        returns the current position's profit
        '''

        return self._profit[-1]

    def update_profit(self, current_price: float):
        '''
        updates the position's current profit

        :args:

        current_price: stock/currency current price to market.
        '''

        if self._option == Options.BUY:
            current_profit = (current_price - self._entry_price) / self._entry_price

        elif self._option == Options.SELL:
            current_profit = (self._entry_price - current_price) / self._entry_price

        self._profit = np.append(self._profit, [current_profit])

    def get_balance(self) -> float:
        '''
        Returns the current position overall balance
        '''

        if self._option == Options.BUY:
            profit_margin = 1 + self._profit[-1]
        else:
            profit_margin = self._profit[-1] #Profit margin calculation are different for short operations
        return self._invested_value * profit_margin

    def get_profit_history(self) -> np.array:
        '''
        Output the positions profit history over all its lifetime.
        
        This method is useful merely for plotting purposes.
        '''

        return self._profit

    def get_final_date(self) -> date:
        '''
        Get position's final date.
        '''
        if not self.is_open():
            raise Exception('This position is still open')
        
        return self._final_date

