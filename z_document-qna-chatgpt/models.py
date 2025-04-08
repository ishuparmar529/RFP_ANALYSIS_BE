from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime, timezone
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    sso = Column(Boolean, default=False)
    role = Column(String, default="user")

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    projects = relationship("Project", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime,  default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey("user.id"))

    chat_history = relationship("ChatHistory", back_populates="project")
    documents = relationship("Document", back_populates="project")
    user = relationship("User", back_populates="projects")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))

    project = relationship("Project", back_populates="documents")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    message = Column(String)
    response = Column(String)
    project_id = Column(Integer, ForeignKey("projects.id"))
    project_name = Column(String)

    created_at = Column(DateTime,  default=lambda: datetime.now(timezone.utc))
    
    project = relationship("Project", back_populates="chat_history")
    user = relationship("User", back_populates="chat_history")