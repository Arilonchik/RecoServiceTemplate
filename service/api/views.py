import os
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKey
from pydantic import BaseModel

from reco_models.model_validator import ModelValidator
from service.api.exceptions import (
    ModelNotFoundError,
    NotAuthorizedError,
    UserNotFoundError,
)
from service.log import app_logger

load_dotenv()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
validator = ModelValidator()

security_scheme = HTTPBearer()
API_KEY = os.getenv("API_KEY")


async def check_key(
    key: HTTPAuthorizationCredentials = Security(security_scheme),
) -> str:
    if key.credentials == API_KEY:
        return key.credentials
    raise NotAuthorizedError()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        404: {"description": "User/model not found"},
        401: {"description": "Authorization error"},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: APIKey = Depends(check_key),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Model reco start
    model = validator.get_reco_model(model_name)
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")
    if model is None:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    reco = model.recommend(user_id)
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
