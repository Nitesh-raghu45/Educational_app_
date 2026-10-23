import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..errors import AppError
from ..models import User
from ..schemas import Credentials, TokenOut, UserOut
from ..security import create_access_token, current_user, hash_password, verify_password
from ..services import usage

router = APIRouter(prefix="/auth", tags=["auth"])


def _token(user: User) -> TokenOut:
    return TokenOut(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


@router.post("/register", response_model=TokenOut, status_code=201)
async def register(body: Credentials, session: AsyncSession = Depends(get_session)) -> TokenOut:
    user = User(id=uuid.uuid4(), email=body.email, password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        raise AppError(409, "EMAIL_TAKEN", "An account with this email already exists.")
    return _token(user)


@router.post("/login", response_model=TokenOut)
async def login(body: Credentials, session: AsyncSession = Depends(get_session)) -> TokenOut:
    user = await session.scalar(select(User).where(User.email == body.email))
    if user is None or not verify_password(body.password, user.password_hash):
        raise AppError(401, "INVALID_CREDENTIALS", "Incorrect email or password.")
    return _token(user)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(current_user)) -> User:
    return user


usage_router = APIRouter(prefix="/usage", tags=["usage"])


@usage_router.get("")
async def my_usage(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)) -> dict:
    """Today's usage against the daily limits (limit 0 = unlimited)."""
    return await usage.get_usage(session, user.id)
