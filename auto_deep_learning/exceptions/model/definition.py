from typing import Optional

class IncorrectCategoryType(Exception):
    def __init__(
        self,
        category_type: Optional[str] = '',
        msg: str = 'Incorrect Category Type'
    ):

        self.msg = str(msg + ': ' + category_type) if category_type else msg
        super().__init__(self.msg)