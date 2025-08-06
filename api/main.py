from fastapi import FastAPI
from .memoryalpha.health import router as health_router
from .memoryalpha.ask import router as ask_router

app = FastAPI()
app.include_router(health_router)
app.include_router(ask_router)