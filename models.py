from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector
from db import engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text)
    embedding = Column(Vector(384))


Base.metadata.create_all(bind=engine)