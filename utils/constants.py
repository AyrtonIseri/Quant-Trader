from enum import Enum

class Options (Enum):
    '''
    Options class. Enumerate all possible options an agent can perform.
    '''
    
    BUY = 1
    SELL = -1
    HOLD = 0

class Positions (Enum):
    '''
    Positions class. Enumerate all possible states a position can be
    '''
    
    OPEN = 1
    CLOSED = 0
    