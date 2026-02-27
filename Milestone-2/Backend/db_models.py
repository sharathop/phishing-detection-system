from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    gmail = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False) # In production, always hash this!

    # Relationship to history
    scans = relationship("ScanHistory", back_populates="owner")

class ScanHistory(Base):
    __tablename__ = "scan_history"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    output = Column(String, nullable=False) # Safe, Suspicious, or Phishing
    time = Column(DateTime, default=datetime.utcnow)
    
    # Link to User
    user_gmail = Column(String, ForeignKey("users.gmail"))
    owner = relationship("User", back_populates="scans")