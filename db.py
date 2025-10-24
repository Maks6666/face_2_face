from sqlalchemy import create_engine, column, integer, string, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.testing.config import db_url

username = "user"
password = 134472
host = "localhost"
port = 5432
database = "detections"

db_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}//{database}"