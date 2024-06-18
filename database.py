from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

URL_DATABASE = 'postgresql://climate_lstm_user:GzS7EJ8Nuuqmq3HQS8NvVzzPRxxKVnRs@dpg-cpoj56ij1k6c73a88q60-a.oregon-postgres.render.com/climate_lstm_db'

engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()