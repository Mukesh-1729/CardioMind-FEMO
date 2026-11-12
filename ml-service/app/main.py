from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import health, predict, federated
from app.services.predictor import load_xgboost_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_xgboost_model()
    yield

app = FastAPI(
    title='CardioMind ML Service',
    version='1.0.0',
    description='FastAPI inference and federated learning service for cardiovascular risk assessment.',
    lifespan=lifespan
)

app.include_router(health.router, prefix='/health', tags=['health'])
app.include_router(predict.router, prefix='/predict', tags=['predict'])
app.include_router(federated.router, prefix='/federated', tags=['federated'])
