"""We could store the response of the forecast classes, response of forecast probabilities. And the correct labels of that forecast.
Jointly with storing the image in the database, as well as saving a date. Could also a new field, which tells if is correct or not.
"""

from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import false
from sqlmodel import Field, SQLModel


class Forecasts(SQLModel, table=True):
    __tablename__ = 'forecasts'

    id: Optional[int] = Field(
        sa_column=sa.Column(
            primary_key=True,
            default=None
        ),
    )
    inference: str = Field(max_length=512, unique=False, nullable=False)
    objective: str = Field(max_length=256)
    correct: bool = Field(sa_column=sa.Column(sa.Boolean, nullable=False, server_default=false()))

    # Which was the throughput & time it took where this inference was made
    throughput: int = Field(sa_column=sa.Column(sa.Integer))
    timing: int = Field(sa_column=sa.Column(sa.Integer))

    created_at: Optional[datetime] = Field(
        sa_column=sa.Column(
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.current_timestamp()
        )
    )
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, nullable=False)
