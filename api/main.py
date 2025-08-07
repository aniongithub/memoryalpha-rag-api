import logging
from fastapi import FastAPI
from .memoryalpha.health import router as health_router
from .memoryalpha.stream import router as stream_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")
    logger.info("Health and stream endpoints available")

app.include_router(health_router)
app.include_router(stream_router)