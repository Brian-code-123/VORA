from pathlib import Path
from typing import Literal

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
    first_chunk_words: int = 1   # Latin words in the first TTS chunk (G2: 1 word p95 148 ms vs 2 words 225 ms on M2)
    second_chunk_words: int = 3  # then 3 words, so chunk 2 is ready before the 1-word chunk 1 finishes playing
    host: str = "127.0.0.1"
    max_sessions: int = 4
    max_frame_bytes: int = 64 * 1024
    idle_timeout_s: float = 60.0
    stall_timeout_s: float = 10.0   # bounded out-queue full this long -> client is not reading, drop the session
    max_text_frame_bytes: int = 4096
    allowed_origins: tuple[str, ...] = ()   # extra allowed Origin values; same-origin (Origin host == Host) is always allowed
    asr_threads: int = 2
    asr_zh_model: Literal["zipformer14m", "ctc_small"] = "ctc_small"   # AISHELL-1 CER 10.9% (wrapper, before no-reset) vs 16.0% for the 14M transducer
    llm_threads: int = 4
    ssl_cert: str = ""
    ssl_key: str = ""
    tts_en_fp32: bool = False  # int8 en voice meets the 30 MB gate but is ~3x slower on ARM/M2; fp32 is 63 MB
