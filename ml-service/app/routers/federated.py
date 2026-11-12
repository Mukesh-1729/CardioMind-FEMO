from fastapi import APIRouter
from app.services.federated import build_federated_status

router = APIRouter()


@router.get('/status')
def federated_status():
    return build_federated_status()
