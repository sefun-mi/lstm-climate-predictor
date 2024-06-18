from sqlalchemy import Boolean, Column, Integer, String, ForeignKey
from database import Base

class PredictionEntity(Base):
    __tablename__ = 'prediction'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    median_tasmax = Column(String, index=True)
    mean_tas = Column(String, index=True)
    median_cwd = Column(String, index=True)
    mean_pr = Column(String, index=True)
    mean_tasmin = Column(String, index=True)
    median_hurs = Column(String, index=True)
    median_tas = Column(String, index=True)
    median_tasmin = Column(String, index=True)
    median_cdd = Column(String, index=True)
    mean_tasmax = Column(String, index=True)

class UserEntity(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    password = Column(String, index=True)
    phonenum = Column(String, index=True)
    is_logged_in = Column(Boolean, index=True)