"""API routes for LC0 Service."""

from fastapi import APIRouter, HTTPException

from ..engine import Lc0Wrapper
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    MoveCandidateResponse,
)

router = APIRouter()

# Engine instance (set by main.py on startup)
_engine: Lc0Wrapper | None = None


def set_engine(engine: Lc0Wrapper) -> None:
    """Set the global engine instance."""
    global _engine
    _engine = engine


def get_engine() -> Lc0Wrapper:
    """Get the engine instance or raise if not available."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_position(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a chess position and return candidate moves with evaluations.

    Uses Leela Chess Zero (LC0) neural network for evaluation.
    """
    engine = get_engine()

    try:
        analysis = await engine.analyze_position(
            fen=request.fen,
            nodes=request.nodes,
            num_moves=request.num_moves,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    candidates = [
        MoveCandidateResponse(
            move=c.move,
            move_san=c.move_san,
            score_cp=c.score_cp,
            score_wdl=list(c.score_wdl),
            rank=c.rank,
        )
        for c in analysis.candidates
    ]

    return AnalyzeResponse(
        fen=analysis.fen,
        candidates=candidates,
        evaluation_cp=analysis.evaluation_cp,
        total_nodes=analysis.total_nodes,
        time_ms=analysis.time_ms,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and engine status."""
    from ..config import settings

    engine_ready = False
    if _engine is not None:
        try:
            engine_ready = await _engine.is_ready()
        except Exception:
            engine_ready = False

    status = "healthy" if engine_ready else "degraded"

    return HealthResponse(
        status=status,
        engine_ready=engine_ready,
        backend=settings.lc0_backend,
        gpu_ids=settings.gpu_ids_list,
    )
