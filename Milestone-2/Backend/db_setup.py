from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# The database file will be created in your root directory
DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_0vM5WjqFLtPr@ep-tiny-bird-aialo8in-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require"

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