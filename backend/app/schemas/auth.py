from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100, examples=["Asha Kumar"])
    email: EmailStr = Field(examples=["asha@example.com"])
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH, examples=["a long passphrase 42"])

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("name must not be blank")
        if any(ord(ch) < 32 for ch in value):
            raise ValueError("name contains control characters")
        return value

    @field_validator("email")
    @classmethod
    def _lower_email(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def _password_not_email(self) -> "RegisterRequest":
        if self.password.lower() == self.email.lower():
            raise ValueError("password must not be the same as the email address")
        return self


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=254, examples=["asha@example.com"])
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)

    @field_validator("email")
    @classmethod
    def _lower_email(cls, value: str) -> str:
        return value.strip().lower()


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Lifetime in seconds")
