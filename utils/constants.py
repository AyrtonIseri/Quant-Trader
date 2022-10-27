from enum import Enum

class Options (Enum):
    '''
    Options class. Enumerate all possible options an agent can perform.
    '''
    
    BUY = 1
    SELL = -1
    HOLD = 0