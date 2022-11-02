from models.Indicators.AI import AI
from torch import nn

class __RNN_module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward():
        pass

class RNN (AI):
    '''
    Class that abstracts the Recurring Neural Network model. Should be inherited by all RNN models.
    '''
    def __init__(self, data_parameters: list[str]):
        super().__init__()
        
        assert(type(data_parameters) == list)
        for parameter in data_parameters:
            assert(type(parameter) == str)

        self._data_parameters = data_parameters