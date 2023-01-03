from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlmodel import Session, SQLModel

ModelType = TypeVar('ModelType', bound=SQLModel)
CreateSchemaType = TypeVar('CreateSchemaType', bound=BaseModel)
UpdateSchemaType = TypeVar('UpdateSchemaType', bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(
        self,
        model: Type[ModelType]
    ):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).
        **Parameters**
        * `model`: A SQLModel model class
        * `schema`: A Pydantic model (schema) class
        """
        self.model = model

    def get(
        self,
        db: Session,
        id: Any
    ) -> Optional[ModelType]:
        """Get a single instance in the database

        Args:
            db (Session): the database in which we want to look for that value
            id (Any): the identification value of that instance

        Returns:
            Optional[ModelType]: _description_
        """

        return db.query(self.model).\
            filter(self.model.id == id).\
            first()

    def get_multi(
        self,
        db: Session,
        *args,
        begin_id: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get a group of instances in the database

        Args:
            db (Session): the database in which we want to look for that value
            begin_id (int, optional): the first identification number. Defaults to 0.
            limit (int, optional): the second identification number. Defaults to 100.

        Returns:
            List[ModelType]: the list of instances that we wanted
        """

        return db.query(self.model).\
            filter(self.model.id >= begin_id).\
            limit(limit).all()

    # TODO: Get with pagination

    def create(
        self,
        db: Session,
        *args,
        obj_in: CreateSchemaType
    ) -> ModelType:
        """Create a new instance in the model

        Args:
            db (Session): the database in which we want to look for that value
            obj_in (CreateSchemaType): the new object taht we want to create

        Returns:
            ModelType: the new instance created in the database
        """

        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *args,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """Update a given instance in the database

        Args:
            db (Session): the database in which we want to look for that value
            db_obj (ModelType): the instance of the object we want to update
            obj_in (Union[UpdateSchemaType, Dict[str, Any]]): the new information we want to update

        Returns:
            ModelType: the db_obj with the data updated
        """

        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)

        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(
        self,
        db: Session,
        *args,
        id: int
    ) -> ModelType:
        """Deletion of an instance in the database

        Args:
            db (Session): the database in which we want to look for that value
            id (int): the identification value of that instance

        Returns:
            ModelType: the instance that we deleted in the database
        """

        obj = db.query(self.model).\
            get(id)

        db.delete(obj)
        db.commit()

        return obj
