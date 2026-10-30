import asyncio
import hashlib
import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..db import get_session
from ..errors import AppError
from ..models import User
from ..ratelimit import client_ip, enforce_limit
from ..schemas import Credentials, TokenOut, UserOut
from ..security import DUMMY_HASH, create_access_token, current_user, hash_password, verify_password
from ..services import usage

router = APIRouter(prefix="/auth", tags=["auth"])


def _token(user: User) -> TokenOut:
    return TokenOut(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


def _email_key(email: str) -> str:
    # Hashed, so rate-limit keys in Redis don't hold email addresses.
    return hashlib.sha256(email.encode()).hexdigest()[:32]


@router.post("/register", response_model=TokenOut, status_code=201)
async def register(body: Credentials, request: Request, session: AsyncSession = Depends(get_session)) -> TokenOut:
    settings = get_settings()
    busy = "Too many new accounts right now. Please try again in a while."
    await enforce_limit(f"signup-ip:{client_ip(request)}", settings.signup_limit_per_ip_per_hour, 3600, busy)
    await enforce_limit("signup-all", settings.signup_limit_per_hour, 3600, busy)
    # bcrypt is deliberately slow (~0.2-0.5 s); in a thread it doesn't stall everyone else's requests.
    password_hash = await asyncio.to_thread(hash_password, body.password)
    user = User(id=uuid.uuid4(), email=body.email, password_hash=password_hash)
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        raise AppError(409, "EMAIL_TAKEN", "An account with this email already exists.")
    return _token(user)


@router.post("/login", response_model=TokenOut)
async def login(body: Credentials, request: Request, session: AsyncSession = Depends(get_session)) -> TokenOut:
    settings = get_settings()
    wait = "Too many sign-in attempts. Please wait 15 minutes and try again."
    await enforce_limit(f"login-ip:{client_ip(request)}", settings.login_limit_per_ip_per_15min, 900, wait)
    # Per account too, which no IP rotation gets around.
    await enforce_limit(f"login-email:{_email_key(body.email)}", settings.login_limit_per_email_per_15min, 900, wait)
    user = await session.scalar(select(User).where(User.email == body.email))
    # Check a password even for unknown emails, so response time doesn't reveal which emails have accounts.
    valid = await asyncio.to_thread(verify_password, body.password, user.password_hash if user else DUMMY_HASH)
    if user is None or not valid:
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
