from typing import Callable
import pandas as pd
from datetime import date

def format_bitcoin_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    formats the specific yahoo finance bitcoin dataframe to the trader format
    '''
    
    dataframe = dataframe.rename(columns={'Date': 'Day', 'Close': 'Price'})
    dataframe.Price = dataframe.Price.convert_dtypes(convert_floating=True)
    
    return dataframe

class Loader():
    '''
    Auxiliary class to load data into the required format
    '''

    def __init__(self):
        '''
        gets today date in required format to load data
        '''
        self._today = date.today().strftime('%d_%m_%Y')

    def load_data(self, df_name:str = None, formatter: Callable = format_bitcoin_dataframe):
        '''
        loads the df_name dataframe and formats it with the 'Price' and 'Day' columns
        '''
        
        if df_name is None:
            df_name = 'bitcoin_historical_data_until_' + self._today

        df = pd.read_csv(df_name, parse_dates = ['Date'], infer_datetime_format = True)

        df = formatter(df)

        return df

