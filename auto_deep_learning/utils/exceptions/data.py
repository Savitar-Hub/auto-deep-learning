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


class ChildFileUnexpected(Exception):
    """It was expected to have a folder with the images values, but instead found a file

    Args:
        Exception : the exception class
    """

    def __init__(
        self,
        path: str,
        msg: str = "Found a file when expecting folder"
    ):

        self.msg = msg + ': ' + path
        super().__init__(self.msg)