from models.Indicators.Indicator import Indicator

class AI (Indicator):
    '''
    Class that abstracts Signals. Should be inherited by all signal models.
    '''
    def __init__(self):
        super().__init__()