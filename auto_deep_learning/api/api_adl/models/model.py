"""Keep track of the models trained and deployed, and which is the path to that file in local (if there is) and in remote (if there is)
"""

from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import false
from sqlmodel import Field, SQLModel


class Models(SQLModel, table=True):
    __tablename__ = 'models'

    id: Optional[int] = Field(
        sa_column=sa.Column(
            primary_key=True,
            default=None
        ),
    )

    # If we want to specify a name to this model
    name: Optional[str] = Field(max_length=256, nullable=True)

    # Where is the path where the model is stored
    path: Optional[str] = Field(max_length=256, nullable=True)

    # Amount of params it has
    size: Optional[int] = Field(sa_column=sa.Column(sa.Integer, nullable=True))

    # The overall accuracy that this model had
    accuracy: Optional[float] = Field(sa_column=sa.Column(sa.Float, nullable=True))

    # Whether this model is selected for production
    correct: bool = Field(sa_column=sa.Column(sa.Boolean, nullable=False, server_default=false()))

    created_at: Optional[datetime] = Field(
        sa_column=sa.Column(
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.current_timestamp()
        )
    )
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, nullable=False)
