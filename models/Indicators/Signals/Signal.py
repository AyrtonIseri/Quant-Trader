from models.Indicators.Indicator import Indicator

class Signal (Indicator):
    '''
    Class that abstracts Signals. Should be inherited by all signal models.
    '''
    def __init__(self):
        super().__init__()
