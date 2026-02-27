from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    gmail: EmailStr
    password: str

class ScanRequest(BaseModel):
    url: str
    gmail: EmailStr

class ScanResponse(BaseModel):
    url: str
    output: str
    time: datetime