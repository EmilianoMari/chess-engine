"""Strutture dati per mosse candidate e analisi posizioni."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MoveCandidate:
    """Rappresenta una mossa candidata con valutazioni da Lc0."""

    move: str  # Notazione UCI (es. "e2e4")
    move_san: str  # Notazione SAN (es. "e4")

    # Valutazioni
    score_cp: int  # Centipawns
    score_wdl: tuple[int, int, int]  # Win/Draw/Loss in millesimi (es. 450, 400, 150)

    # Informazioni ricerca
    pv: list[str] = field(default_factory=list)  # Principal variation in UCI
    pv_san: list[str] = field(default_factory=list)  # Principal variation in SAN
    nodes: int = 0  # Nodi esplorati per questa mossa
    depth: int = 0  # Profondità raggiunta

    # Policy dalla rete neurale
    policy: float = 0.0  # Probabilità (0.0 - 1.0)

    # Metadata
    rank: int = 0  # Posizione nel ranking (1 = migliore)
    multipv_index: int = 0  # Indice MultiPV

    def __post_init__(self) -> None:
        """Validazione post-inizializzazione."""
        if self.score_wdl:
            assert len(self.score_wdl) == 3, "WDL deve avere 3 valori"
            assert sum(self.score_wdl) == 1000, "WDL deve sommare a 1000"

    @property
    def win_probability(self) -> float:
        """Probabilità di vittoria (0.0 - 1.0)."""
        return self.score_wdl[0] / 1000.0

    @property
    def draw_probability(self) -> float:
        """Probabilità di patta (0.0 - 1.0)."""
        return self.score_wdl[1] / 1000.0

    @property
    def loss_probability(self) -> float:
        """Probabilità di sconfitta (0.0 - 1.0)."""
        return self.score_wdl[2] / 1000.0

    @property
    def expected_score(self) -> float:
        """Punteggio atteso (0.0 - 1.0), dove 1.0 = vittoria certa."""
        return self.win_probability + 0.5 * self.draw_probability

    def to_dict(self) -> dict:
        """Converte in dizionario per serializzazione JSON."""
        return {
            "move": self.move,
            "move_san": self.move_san,
            "score_cp": self.score_cp,
            "score_wdl": list(self.score_wdl),
            "pv": self.pv,
            "pv_san": self.pv_san,
            "nodes": self.nodes,
            "depth": self.depth,
            "policy": self.policy,
            "rank": self.rank,
            "win_probability": self.win_probability,
            "draw_probability": self.draw_probability,
            "loss_probability": self.loss_probability,
            "expected_score": self.expected_score,
        }


@dataclass
class PositionAnalysis:
    """Analisi completa di una posizione."""

    fen: str
    candidates: list[MoveCandidate]

    # Valutazione globale posizione
    evaluation_cp: int = 0
    evaluation_wdl: tuple[int, int, int] = (333, 334, 333)

    # Statistiche ricerca
    total_nodes: int = 0
    time_ms: int = 0
    nps: int = 0  # Nodi per secondo

    # Metadati
    depth: int = 0
    seldepth: int = 0
    multipv: int = 1

    @property
    def best_move(self) -> MoveCandidate | None:
        """Ritorna la mossa migliore."""
        return self.candidates[0] if self.candidates else None

    @property
    def is_winning(self) -> bool:
        """True se la posizione è chiaramente vincente (>= +2.0)."""
        return self.evaluation_cp >= 200

    @property
    def is_losing(self) -> bool:
        """True se la posizione è chiaramente perdente (<= -2.0)."""
        return self.evaluation_cp <= -200

    @property
    def game_phase(self) -> Literal["opening", "middlegame", "endgame"]:
        """Stima la fase della partita basandosi sul materiale."""
        total_moves = len(self.candidates)
        if total_moves > 25:
            return "opening"
        elif total_moves > 10:
            return "middlegame"
        return "endgame"

    def to_dict(self) -> dict:
        """Converte in dizionario per serializzazione JSON."""
        return {
            "fen": self.fen,
            "candidates": [c.to_dict() for c in self.candidates],
            "evaluation_cp": self.evaluation_cp,
            "evaluation_wdl": list(self.evaluation_wdl),
            "total_nodes": self.total_nodes,
            "time_ms": self.time_ms,
            "nps": self.nps,
            "depth": self.depth,
            "seldepth": self.seldepth,
        }
