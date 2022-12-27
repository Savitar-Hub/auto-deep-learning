from typing import Optional


class InvalidArgumentType(Exception):
    def __init__(
        self,
        variable_name: str,
        variable_expected_type: str,
        msg: str = 'Invalid Argument Type'
    ):

        self.msg: str = msg + f': for variable, {variable_name}, expected the type of; {variable_expected_type}'

        super().__init__(self.msg)


class InconsistentInput(Exception):
    def __init__(
        self,
        variable_name: Optional[str] = '',
        msg: str = 'Inconsistent variable input'
    ):

        self.msg = str(msg + ': ' + variable_name) if variable_name else msg
        super().__init__(self.msg)