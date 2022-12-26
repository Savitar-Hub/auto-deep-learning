class ImbalancedError(Exception):
    """Error when we have some classes that are not on the other split types."""
    
    def __init__(
        self,
        msg
    ):

        self.msg = msg
        super().__init__(self.msg)