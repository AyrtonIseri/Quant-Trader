from Indicator import Indicator
from models.Model import Model

class Signal (Indicator):
    '''
    Class that abstracts Signals. Should be inherited by all signal models.
    '''
    def __init__(self, model: Model):
        super().__init__(model)
