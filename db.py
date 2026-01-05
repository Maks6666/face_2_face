from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.testing.config import db_url

from personal_data import dict

username = dict["username"]
password = dict["passrord"]
host = "localhost"
port = 5432
database = dict["db_name"]

db_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(db_url)
Base = declarative_base()

class Table(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True)
    idx = Column(Integer)
    action = Column(String)
    detected_at = Column(DateTime, default=func.now())

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()


