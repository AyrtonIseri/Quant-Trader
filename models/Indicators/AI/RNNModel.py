from models.Indicators.AI import AI

class RNN (AI):
    '''
    Class that abstracts the Recurring Neural Network model. Should be inherited by all RNN models.
    '''
    def __init__(self):
        super().__init__()