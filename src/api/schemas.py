"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""

    fen: str = Field(
        ...,
        description="Chess position in FEN notation",
        examples=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
    )
    nodes: int = Field(
        default=100000,
        ge=1000,
        le=10000000,
        description="Number of nodes to search",
    )
    num_moves: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of candidate moves to return",
    )


class MoveCandidateResponse(BaseModel):
    """A single move candidate with evaluation."""

    move: str = Field(..., description="Move in UCI notation (e.g., 'e2e4')")
    move_san: str = Field(..., description="Move in SAN notation (e.g., 'e4')")
    score_cp: int = Field(..., description="Score in centipawns")
    score_wdl: list[int] = Field(
        ..., description="Win/Draw/Loss probabilities (per mille)"
    )
    rank: int = Field(..., description="Rank of the move (1 = best)")

    class Config:
        json_schema_extra = {
            "example": {
                "move": "e2e4",
                "move_san": "e4",
                "score_cp": 35,
                "score_wdl": [450, 500, 50],
                "rank": 1,
            }
        }


class AnalyzeResponse(BaseModel):
    """Response body for /analyze endpoint."""

    fen: str = Field(..., description="The analyzed position")
    candidates: list[MoveCandidateResponse] = Field(
        ..., description="Ranked list of candidate moves"
    )
    evaluation_cp: int = Field(..., description="Position evaluation in centipawns")
    total_nodes: int = Field(..., description="Total nodes searched")
    time_ms: int = Field(..., description="Time spent analyzing in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "candidates": [
                    {
                        "move": "e2e4",
                        "move_san": "e4",
                        "score_cp": 35,
                        "score_wdl": [450, 500, 50],
                        "rank": 1,
                    }
                ],
                "evaluation_cp": 35,
                "total_nodes": 100000,
                "time_ms": 2500,
            }
        }


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field(..., description="Service status")
    engine_ready: bool = Field(..., description="Whether LC0 engine is ready")
    backend: str = Field(..., description="LC0 backend in use")
    gpu_ids: list[int] = Field(..., description="GPU IDs being used")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "engine_ready": True,
                "backend": "cuda-fp16",
                "gpu_ids": [0],
            }
        }
