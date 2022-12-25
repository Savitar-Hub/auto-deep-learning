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


class InvalidSplitType(Exception):
    """Invalid Split Type {train, valid, test}

    Args:
        Exception : the exception class
    """

    def __init__(
        self,
        invalid_split_name: str,
        msg: str = 'Invalid Split Type'
    ):
        self.msg = msg + ': ' + invalid_split_name
        self.msg += '\nShould be one of: {train, valid, test}'

        super().__init__(self.msg)


class InvalidFileExtension(Exception):
    """Found a file that was not an image

    Args:
        Exception : the exception class
    """

    def __init__(
        self,
        image_path: str,
        msg: str = 'Found file that is not an image extension supported'
    ):

        self.msg = msg + ': ' + image_path
        super().__init__(self.msg)