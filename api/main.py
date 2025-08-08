import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .memoryalpha.health import router as health_router
from .memoryalpha.stream import router as stream_router
from .memoryalpha.ask import router as ask_router
from .memoryalpha.identify import router as identify_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI application starting up...")
    logger.info("Health and stream endpoints available")
    yield
    # Shutdown (if needed)
    logger.info("FastAPI application shutting down...")

app = FastAPI(lifespan=lifespan)

app.include_router(health_router)
app.include_router(stream_router)
app.include_router(ask_router)
app.include_router(identify_router)