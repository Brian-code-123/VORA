from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VORA_")

    models_dir: Path = ROOT / "models"
    index_dir: Path = ROOT / "index"
    kb_dir: Path = ROOT / "kb"
    asr_chunk_ms: int = 320
    top_k: int = 3
    min_score: float = 0.35  # calibrated in Task 5 (grid search on dev split)
    max_ctx_tokens: int = 120
    queue_max: int = 8
    filler: bool = False
    host: str = "127.0.0.1"
    max_sessions: int = 4
    max_frame_bytes: int = 64 * 1024
    idle_timeout_s: float = 60.0
    asr_threads: int = 2
    llm_threads: int = 4
    tts_en_fp32: bool = False  # int8 en voice meets the 30 MB gate but is ~3x slower on ARM/M2; fp32 is 63 MB
