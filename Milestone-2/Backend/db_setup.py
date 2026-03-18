from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# The database file will be created in your root directory
DATABASE_URL = os.getenv("DATABASE_URL")

# connect_args is needed only for SQLite
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()