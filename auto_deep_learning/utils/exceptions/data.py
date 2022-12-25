from typing import Optional, List


class NoFolderData(Exception):
    """No folders found in the path provided

    Args:
        Exception: the exception class
    """

    def __init__(
        self, 
        msg: str = "No folders found in the path provided", 
    ):

        self.msg = msg
        super().__init__(self.msg)