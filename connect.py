from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from personal_data import dict

username = dict["username"]
# add text file with your DB password as a single string
password = dict["passrord"]
host = "localhost"
port = 5432
database_name = dict["db_name"]

db_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database_name}"

try:
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    Base = declarative_base()
    print("Connected to DB!")
except Exception as e:
    print(f"Connection error: {e}")


class Table(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True)
    idx = Column(String)
    age = Column(String)
    emotion = Column(String)
    gender = Column(String)
    race = Column(String)
    detected_at = Column(DateTime, default=func.now())


Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# row = Table(idx=2, age="Adult", emotion="Neutral", gender="male", race="white")
# session.add(row)
# session.commit()