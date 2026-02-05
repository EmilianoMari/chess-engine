"""Wrapper per Leela Chess Zero (Lc0) via protocollo UCI."""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

import chess
import structlog

from .move_candidates import MoveCandidate, PositionAnalysis

logger = structlog.get_logger(__name__)


@dataclass
class Lc0Config:
    """Configurazione per Lc0 engine."""

    executable_path: Path
    network_path: Path
    backend: str = "cuda-fp16"
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    hash_mb: int = 4096
    threads: int = 4

    # Parametri MCTS
    minibatch_size: int = 256
    max_prefetch: int = 32
    cpuct: float = 1.745
    fpu_strategy: str = "reduction"
    fpu_value: float = 0.33

    # MultiPV
    multipv: int = 10

    def to_uci_options(self) -> list[tuple[str, str]]:
        """Converte config in opzioni UCI."""
        gpu_str = ",".join(str(g) for g in self.gpu_ids)
        return [
            ("WeightsFile", str(self.network_path)),
            ("Backend", self.backend),
            ("BackendOptions", f"gpu={gpu_str}"),
            ("Hash", str(self.hash_mb)),
            ("Threads", str(self.threads)),
            ("MinibatchSize", str(self.minibatch_size)),
            ("MaxPrefetch", str(self.max_prefetch)),
            ("CPuct", str(self.cpuct)),
            ("FpuStrategy", self.fpu_strategy),
            ("FpuValue", str(self.fpu_value)),
            ("MultiPV", str(self.multipv)),
            ("VerboseMoveStats", "true"),
            ("LogLiveStats", "true"),
            ("SmartPruningFactor", "0"),  # Disabilita pruning per analisi completa
        ]


class Lc0Wrapper:
    """Wrapper asincrono per comunicazione con Lc0 via UCI."""

    def __init__(self, config: Lc0Config):
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self._is_ready = False

    async def start(self) -> None:
        """Avvia il processo Lc0."""
        if self._process is not None:
            logger.warning("Lc0 già avviato, riavvio...")
            await self.stop()

        logger.info("Avvio Lc0", path=str(self.config.executable_path))

        self._process = await asyncio.create_subprocess_exec(
            str(self.config.executable_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._reader = self._process.stdout
        self._writer = self._process.stdin

        # Inizializzazione UCI
        await self._send_command("uci")
        await self._wait_for("uciok")

        # Configura opzioni
        for name, value in self.config.to_uci_options():
            await self._send_command(f"setoption name {name} value {value}")

        # Prepara engine
        await self._send_command("isready")
        await self._wait_for("readyok")

        self._is_ready = True
        logger.info("Lc0 pronto", backend=self.config.backend, gpus=self.config.gpu_ids)

    async def stop(self) -> None:
        """Ferma il processo Lc0."""
        if self._process is None:
            return

        logger.info("Arresto Lc0")
        await self._send_command("quit")

        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout arresto Lc0, terminazione forzata")
            self._process.kill()
            await self._process.wait()

        self._process = None
        self._reader = None
        self._writer = None
        self._is_ready = False

    async def _send_command(self, command: str) -> None:
        """Invia comando UCI a Lc0."""
        if self._writer is None:
            raise RuntimeError("Lc0 non avviato")

        logger.debug("UCI >>", command=command)
        self._writer.write(f"{command}\n".encode())
        await self._writer.drain()

    async def _read_line(self) -> str:
        """Legge una linea da Lc0."""
        if self._reader is None:
            raise RuntimeError("Lc0 non avviato")

        line = await self._reader.readline()
        decoded = line.decode().strip()
        if decoded:
            logger.debug("UCI <<", response=decoded)
        return decoded

    async def _wait_for(self, expected: str) -> list[str]:
        """Legge linee fino a trovare quella attesa."""
        lines = []
        while True:
            line = await self._read_line()
            lines.append(line)
            if line.startswith(expected):
                return lines

    async def _read_until_bestmove(self) -> tuple[list[str], str]:
        """Legge output fino a bestmove, ritorna tutte le linee info."""
        info_lines = []
        while True:
            line = await self._read_line()
            if line.startswith("info"):
                info_lines.append(line)
            elif line.startswith("bestmove"):
                return info_lines, line

    async def analyze_position(
        self,
        fen: str,
        nodes: int | None = None,
        depth: int | None = None,
        time_ms: int | None = None,
        num_moves: int | None = None,
    ) -> PositionAnalysis:
        """
        Analizza una posizione e ritorna le mosse candidate.

        Args:
            fen: Posizione in formato FEN
            nodes: Numero di nodi da esplorare
            depth: Profondità massima
            time_ms: Tempo massimo in millisecondi
            num_moves: Numero di mosse candidate (override MultiPV)
        """
        async with self._lock:
            if not self._is_ready:
                raise RuntimeError("Lc0 non pronto, chiamare start() prima")

            # Imposta MultiPV se richiesto
            if num_moves is not None and num_moves != self.config.multipv:
                await self._send_command(f"setoption name MultiPV value {num_moves}")

            # Imposta posizione
            await self._send_command(f"position fen {fen}")

            # Costruisci comando go
            go_cmd = "go"
            if nodes is not None:
                go_cmd += f" nodes {nodes}"
            if depth is not None:
                go_cmd += f" depth {depth}"
            if time_ms is not None:
                go_cmd += f" movetime {time_ms}"
            if go_cmd == "go":
                # Default: usa nodes dalla config
                go_cmd += f" nodes {100000}"

            await self._send_command(go_cmd)

            # Leggi risultati
            info_lines, bestmove_line = await self._read_until_bestmove()

            # Ripristina MultiPV se modificato
            if num_moves is not None and num_moves != self.config.multipv:
                await self._send_command(
                    f"setoption name MultiPV value {self.config.multipv}"
                )

            # Parse risultati
            return self._parse_analysis(fen, info_lines, bestmove_line)

    def _parse_analysis(
        self, fen: str, info_lines: list[str], bestmove_line: str
    ) -> PositionAnalysis:
        """Parsa l'output UCI in strutture dati."""
        board = chess.Board(fen)
        candidates: dict[int, MoveCandidate] = {}

        total_nodes = 0
        time_ms = 0
        nps = 0
        max_depth = 0
        max_seldepth = 0

        # Regex per parsing info lines
        multipv_pattern = re.compile(r"multipv (\d+)")
        depth_pattern = re.compile(r" depth (\d+)")
        seldepth_pattern = re.compile(r"seldepth (\d+)")
        nodes_pattern = re.compile(r"nodes (\d+)")
        time_pattern = re.compile(r" time (\d+)")
        nps_pattern = re.compile(r"nps (\d+)")
        score_cp_pattern = re.compile(r"score cp (-?\d+)")
        score_mate_pattern = re.compile(r"score mate (-?\d+)")
        wdl_pattern = re.compile(r"wdl (\d+) (\d+) (\d+)")
        pv_pattern = re.compile(r" pv (.+)$")

        for line in info_lines:
            if "multipv" not in line:
                continue

            # Estrai multipv index
            mpv_match = multipv_pattern.search(line)
            if not mpv_match:
                continue
            mpv_idx = int(mpv_match.group(1))

            # Estrai profondità
            depth_match = depth_pattern.search(line)
            depth = int(depth_match.group(1)) if depth_match else 0

            seldepth_match = seldepth_pattern.search(line)
            seldepth = int(seldepth_match.group(1)) if seldepth_match else 0

            # Estrai nodi/tempo
            nodes_match = nodes_pattern.search(line)
            nodes = int(nodes_match.group(1)) if nodes_match else 0

            time_match = time_pattern.search(line)
            if time_match:
                time_ms = max(time_ms, int(time_match.group(1)))

            nps_match = nps_pattern.search(line)
            if nps_match:
                nps = max(nps, int(nps_match.group(1)))

            total_nodes = max(total_nodes, nodes)
            max_depth = max(max_depth, depth)
            max_seldepth = max(max_seldepth, seldepth)

            # Estrai score
            score_cp = 0
            cp_match = score_cp_pattern.search(line)
            mate_match = score_mate_pattern.search(line)

            if mate_match:
                mate_in = int(mate_match.group(1))
                score_cp = 10000 - abs(mate_in) if mate_in > 0 else -10000 + abs(mate_in)
            elif cp_match:
                score_cp = int(cp_match.group(1))

            # Estrai WDL
            wdl = (333, 334, 333)  # Default
            wdl_match = wdl_pattern.search(line)
            if wdl_match:
                wdl = (
                    int(wdl_match.group(1)),
                    int(wdl_match.group(2)),
                    int(wdl_match.group(3)),
                )

            # Estrai PV
            pv: list[str] = []
            pv_san: list[str] = []
            pv_match = pv_pattern.search(line)
            if pv_match:
                pv = pv_match.group(1).split()
                # Converti in SAN
                temp_board = board.copy()
                for move_uci in pv:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in temp_board.legal_moves:
                            pv_san.append(temp_board.san(move))
                            temp_board.push(move)
                        else:
                            break
                    except (ValueError, chess.InvalidMoveError):
                        break

            # Crea/aggiorna candidato
            if pv:
                move_uci = pv[0]
                try:
                    move = chess.Move.from_uci(move_uci)
                    move_san = board.san(move)
                except (ValueError, chess.InvalidMoveError):
                    continue

                candidates[mpv_idx] = MoveCandidate(
                    move=move_uci,
                    move_san=move_san,
                    score_cp=score_cp,
                    score_wdl=wdl,
                    pv=pv,
                    pv_san=pv_san,
                    nodes=nodes,
                    depth=depth,
                    policy=0.0,
                    rank=mpv_idx,
                    multipv_index=mpv_idx,
                )

        # Ordina candidati per rank
        sorted_candidates = [
            candidates[k] for k in sorted(candidates.keys()) if k in candidates
        ]

        # Valutazione globale dalla prima mossa
        evaluation_cp = sorted_candidates[0].score_cp if sorted_candidates else 0
        evaluation_wdl = sorted_candidates[0].score_wdl if sorted_candidates else (333, 334, 333)

        return PositionAnalysis(
            fen=fen,
            candidates=sorted_candidates,
            evaluation_cp=evaluation_cp,
            evaluation_wdl=evaluation_wdl,
            total_nodes=total_nodes,
            time_ms=time_ms,
            nps=nps,
            depth=max_depth,
            seldepth=max_seldepth,
            multipv=len(sorted_candidates),
        )

    async def get_best_move(
        self, fen: str, nodes: int | None = None, time_ms: int | None = None
    ) -> str | None:
        """Ritorna solo la mossa migliore (più veloce di analyze_position)."""
        analysis = await self.analyze_position(fen, nodes=nodes, time_ms=time_ms, num_moves=1)
        return analysis.best_move.move if analysis.best_move else None

    async def is_ready(self) -> bool:
        """Verifica se l'engine è pronto."""
        if not self._is_ready:
            return False

        async with self._lock:
            await self._send_command("isready")
            await self._wait_for("readyok")
            return True

    async def new_game(self) -> None:
        """Prepara l'engine per una nuova partita."""
        async with self._lock:
            await self._send_command("ucinewgame")
            await self._send_command("isready")
            await self._wait_for("readyok")

    @property
    def is_running(self) -> bool:
        """True se il processo Lc0 è in esecuzione."""
        return self._process is not None and self._process.returncode is None


async def create_engine(config: Lc0Config) -> Lc0Wrapper:
    """Factory function per creare e avviare un engine."""
    engine = Lc0Wrapper(config)
    await engine.start()
    return engine
