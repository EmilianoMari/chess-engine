"""LC0 Service - FastAPI entrypoint."""

import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router, set_engine
from .config import settings
from .engine import Lc0Config, Lc0Wrapper

# Configure logging
log_level = "DEBUG" if settings.debug else "INFO"
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.stdlib.logging.INFO if not settings.debug else structlog.stdlib.logging.DEBUG
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global engine instance
_engine: Lc0Wrapper | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - starts and stops LC0 engine."""
    global _engine

    logger.info(
        "Starting LC0 Service",
        lc0_path=str(settings.lc0_path),
        network=str(settings.lc0_network),
        backend=settings.lc0_backend,
        gpu_ids=settings.gpu_ids_list,
    )

    # Create engine config
    config = Lc0Config(
        executable_path=settings.lc0_path,
        network_path=settings.lc0_network,
        backend=settings.lc0_backend,
        gpu_ids=settings.gpu_ids_list,
        hash_mb=settings.lc0_hash_mb,
        threads=settings.lc0_threads,
        multipv=settings.default_num_moves,
    )

    # Start engine
    _engine = Lc0Wrapper(config)
    try:
        await _engine.start()
        set_engine(_engine)
        logger.info("LC0 engine started successfully")
    except Exception as e:
        logger.error("Failed to start LC0 engine", error=str(e))
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Shutting down LC0 Service")
    if _engine is not None:
        await _engine.stop()
        logger.info("LC0 engine stopped")


# Create FastAPI app
app = FastAPI(
    title="LC0 Service",
    description="Standalone Leela Chess Zero engine service for chess position analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
