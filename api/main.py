from fastapi import FastAPI
from .memoryalpha.health import router as health_router
from .memoryalpha.stream import router as stream_router

app = FastAPI()
app.include_router(health_router)
app.include_router(stream_router)