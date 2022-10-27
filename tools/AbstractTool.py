
class AbstractTool:
    '''
    Abstract class that all tools should inherit to work with agents and perform basic tasks over them

    tasks performed by tools could be:
        -Performance evaluation
        -Identifying most efficient model
        -Suggesting general enhancements to the agent (such as rebalancing models share on wallet)
    '''
    
    def __init__(self):
        pass