from  datetime import datetime
from tools import get_pass
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

username = "postgres"
# add text file with your DB password as a single string
password = get_pass("password.txt")
host = "localhost"
port = 5432
database_name = "detections"

db_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database_name}"

try:
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    Base = declarative_base()
    print("Connected to DB!")
except Exception as e:
    print(f"Connection error: {e}")

# Long-term info table
class Table(Base):
    __tablename__ = 'object_detection'
    object_id = Column(Integer, primary_key=True, index=True)
    tracked_object_id = Column(Integer, nullable=False)
    object_class = Column(String, nullable=False)
    age_category = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    race = Column(String, nullable=False)
    face = Column(String, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# short term info table
class IndexTable(Base):
    __tablename__ = 'object_detection_idx'

    object_id_idx = Column(Integer, primary_key=True, index=True)
    tracked_object_idx = Column(Integer, nullable=False)
    age_category_idx = Column(Integer, nullable=False)
    gender_idx = Column(Integer, nullable=False)
    race_idx = Column(Integer, nullable=False)
    face_idx = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    total_obj = Column(Integer, nullable=False)
    time_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# class TotalIndexTable(Base):
#     __tablename__ = 'total_objects'
#     object_id_ = Column(Integer, primary_key=True, index=True)
#     status = Column(String, nullable=False)
#     total_obj = Column(Integer, nullable=False)
#     time_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)



Base.metadata.create_all(bind=engine)

def get_data(session, object_id, tracked_object_id, obj_class, new_age, new_gender, new_race, new_face):
    object_id = int(object_id)

    # literally
    # SELECT * FROM AgeTable
    # WHERE object_id = object_id
    # it will take first selected value and put it into 'obj' by .first()

    obj = session.query(Table).filter_by(object_id=object_id).first()
    # if mentioned id exists
    if obj:
        if new_age is not None:
            obj.age_category = new_age
        if new_gender is not None:
            obj.gender = new_gender
        if new_race is not None:
            obj.race = new_race
        if new_face is not None:
            obj.face = new_face

        obj.last_updated = datetime.utcnow()
        session.commit()

        # return obj.age_category, obj.gender
    # if it doesnt exist - it creates new row in database
    else:

        if new_age is None:
            new_age = 'Unknown'

        if new_gender is None:
            new_gender = 'Unknown'

        if new_race is None:
            new_race = 'Unknown'

        if new_face is None:
            new_face = 'Unknown'

        # if new_age != 'Unknown' and new_gender != 'Unknown' and new_race != 'Unknown':
        if new_age is not None and new_gender is not None  and new_race is not None and new_face is not None:
            obj = Table(object_id=object_id, tracked_object_id = tracked_object_id, object_class = obj_class, age_category=new_age, gender = new_gender, race = new_race, face = new_face)
            session.add(obj)
            session.commit()
        # return obj.age_category


def get_idx(session, object_id_idx, tracked_object_idx, age_category_idx, gender_idx, race_idx, face_idx, total_obj, status='detected'):
    object_id_idx = int(object_id_idx)
    obj = (session.query(IndexTable).filter_by(object_id_idx=object_id_idx)
           .order_by(IndexTable.time_updated.desc()).first())

    if obj:
        if age_category_idx is not None:
            obj.age_category = age_category_idx
        if gender_idx is not None:
            obj.gender = gender_idx
        if race_idx is not None:
            obj.race = race_idx
        if face_idx is not None:
            obj.face = face_idx

        obj.total_obj = total_obj
        obj.last_updated = datetime.utcnow()
        session.commit()

    else:

        if age_category_idx is None:
            age_category_idx = -1

        if gender_idx is None:
            gender_idx = -1

        if race_idx is None:
            race_idx = -1

        if face_idx is None:
            face_idx = -1

        # if new_age != 'Unknown' and new_gender != 'Unknown' and new_race != 'Unknown':
        if age_category_idx is not None and gender_idx is not None  and race_idx is not None and face_idx is not None:
            obj = IndexTable(object_id_idx=object_id_idx, tracked_object_idx = tracked_object_idx, age_category_idx = int(age_category_idx), gender_idx = int(gender_idx), race_idx = int(race_idx),
                             face_idx = int(face_idx), total_obj = total_obj, status = status)
            session.add(obj)
            session.commit()




def detected_filter(session, objects_to_delete, detected_idx):
    t_obj = session.query(IndexTable.object_id_idx).all()
    for obj in t_obj:
        obj = int(obj.object_id_idx)
        if obj not in detected_idx:
            object_to_delete = session.query(IndexTable).filter_by(object_id_idx=obj).first()
            object_to_delete.status = 'not_detected'
            session.commit()

    # print(f"Detected: {detected_idx}")
    #
    # for obj in objects_to_delete:
    #     session.delete(obj)
    #     session.commit()