from typing import List

class IncorrectFolderStructure(Exception):
    """Expected folders of train/valid/test."""

    def __init__(
        self, 
        folder_structure: List[str] = [],
        msg: str = 'Expected structure of train and valid/test (at least one of those two)'
    ) -> None:

        self.folder_structure: str = '/'.join(folder_structure)
        self.msg = msg + f': {self.folder_structure}' if self.folder_structure else msg

        super().__init__(self.msg)


class ImbalancedClassError(Exception):
    """Error when we have some classes that are not on the other split types."""
    
    def __init__(
        self,
        msg: str = 'There are classes in certain split types that are not in the others'
    ):

        self.msg = msg
        super().__init__(self.msg)