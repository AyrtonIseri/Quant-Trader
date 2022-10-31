from datetime import date
from Indicators.Indicator import Indicator
from utils.position import Position, Options
import pandas as pd
import numpy as np

class Model:
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
        self._invested_capital = np.array([0])
        self._take_profit = take_profit
        self._stop_loss = stop_loss
        self._percentage_to_invest = invest_percentage
        self._no_of_positions = 0
        self._indicator = indicator
        self._data = historical_data


    def _create_position(self, price: float, value_to_invest:float, date: date, option: Options):
        '''
        Creates a new position to bet in the current model

        :args:

        price: Price in which the new position was made.
        value_to_invest: amount of money betted in the position
        date: day of the position's creation
        option: Enum option (BUY/SELL/HOLD)
        '''

        new_position = Position(entry_price=price, invested_value=value_to_invest, initial_date=date, option=option)
        self._positions.append(new_position)
        self._no_of_positions += 1


    def _end_position(self, position: Position, date: date):
        '''
        Close an opened position.

        :args:

        position: position to end
        date: final date of the position

        '''
        if position.is_open():
            position.close_position(final_date = date)
        else:
            raise Exception(f'Current position has already been closed.')


    def _update_balance(self, account_variation: float):
        '''
        more than one transaction can be made per day. Therefore, here we will use the net of all daily transactions

        :args:

        account_variation: sum of total inflows and outflows of balance.
        '''
        last_balance = self._account_balance[-1]
        self._account_balance = np.append(self._account_balance, [last_balance + account_variation])
        

    def _set_total_investments(self, total_investments: float):
        '''
        more than one transaction can be made per day. Therefore, here we will use the net of all daily transactions

        :args:

        account_variation: sum of total variations of investments + realized investments.
        '''
        self._invested_capital = np.append(self._invested_capital, [total_investments])
        

    def _position_should_end(self, position: Position) -> bool:
        '''
        Evaluate whether a specific position has reached its end

        :args:

        position: Position to evaluate its continuity
        '''

        current_profit = position.get_profit()
        
        if (current_profit > self._take_profit) or (current_profit < self._stop_loss):
            return True
        return False


    def _get_daily_option(self, current_date: date) -> Options:
        '''
        Calls the indicator and gets the daily option.

        :args:
        current_date: date to call the Indicator on.
        '''

        return self._indicator.get_option(self._data, current_date)


    def _daily_loop(self, current_date: date, current_price: float):
        '''
        Defines the daily from getting the data option to updating all Model parameters

        :args:
        current_date: date to perform the loop.
        current_price: current stock/currency price
        '''
        
        realized_investments = 0 #net investments that had their position closed
        new_investments = 0 #new investments made using the balance account
        investments_balance = 0 #update summation of investments balance

        """
        Ultimately:
        new investments total account: summation (investments_balance)
        balance account variance: realized investments - new investments
        """

        for position in self._positions:
            if position.is_open():
                position.update_profit(current_price = current_price)

                if self._position_should_end(position=position):
                    self._end_position(position=position, date=current_date)
                    realized_investments += position.get_balance()  #Every ended position will get money back to the account

                else:
                    #Every not ended position should have its balance updated in the investments account
                    investments_balance += position.get_balance()
        
        new_option = self._get_daily_option(current_date=current_date)

        if new_option != Options.HOLD:
            new_investments += self._account_balance * self._percentage_to_invest
            self._create_position(price=current_price, date=current_date, 
                                  option=new_option, value_to_invest=new_investments)

        if new_option == Options.SELL:
            new_investments = 0 #short options don't take money to apply since they're leveraged

        account_variance = realized_investments - new_investments
        investments_balance += new_investments

        self._update_balance(account_variation=account_variance)
        self._set_total_investments(investments_balance)
    

    def run_model(self):
        '''
        runs the model over all the data entry. Results can be accessed by getter methods.

        Once again, it's really important that the 'data' propriety be in the correct format
        (idx, Day, Price) at least
        '''

        for idx in self._data.index:
            day = self._data.Day[idx]
            price = self._data.Price[idx]

            self._daily_loop(day, price)



    def get_daily_results(self) -> pd.DataFrame:
        '''
        Returns both account and investments balance in a pd.DataFrame format
        '''

        result = pd.DataFrame({'Day': self._data.Day,
                               'account_balance': self._account_balance,
                               'investments_balance': self._invested_capital})

        return result


    def get_account_balance(self) -> float:
        '''
        retrieves the latest account balance registered in the model
        '''
        return self._account_balance[-1]


    def get_investments_value(self) -> float:
        '''
        returns the final invested value registered in the model
        '''
        return self._invested_capital[-1]


    def get_total_balance(self) -> float:
        '''
        returns the final balance the model accomplished
        '''
        account_balance = self.get_account_balance()
        investment_balance = self.get_investments_value()

        return account_balance + investment_balance