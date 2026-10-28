import asyncio
import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..errors import AppError
from ..figure_store import figure_store
from ..models import User
from ..security import current_user
from ..services.figures import get_owned_figure

router = APIRouter(prefix="/figures", tags=["figures"])


@router.get("/{figure_id}")
async def get_figure(
    figure_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> Response:
    """The figure image, only for the student who owns the document it came from."""
    figure = await get_owned_figure(session, figure_id, user.id)
    data = await asyncio.to_thread(figure_store().get, figure.storage_key)
    if data is None:
        raise AppError(404, "FIGURE_NOT_FOUND", "This figure's image is no longer stored. Upload the document again.")
    # Images never change for an id (re-processing makes new ids), so browsers may keep them.
    return Response(content=data, media_type="image/webp", headers={"Cache-Control": "private, max-age=86400"})
