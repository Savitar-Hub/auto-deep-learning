


class InvalidArgumentType(Exception):
    def __init__(
        self,
        variable_name: str,
        variable_expected_type: str,
        msg: str = 'Invalid Argument Type'
    ):

        self.msg: str = msg + f': for variable, {variable_name}, expected the type of; {variable_expected_type}'

        super().__init__(self.msg)