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
    min_score: float = 0.35  # fallback only: the index meta carries a per-language threshold calibrated on dev data
    dense_weight: float = 0.85       # retrieval score = w * dense + (1-w) * bm25 (dense-only measured about 1 pt better on two KBs; the BM25 share keeps rare tokens such as model codes matchable in custom KBs)
    passage_weight: float = 0.5      # dense score = (1-w) * chunk similarity + w * best sentence similarity (0 = chunk only)
    context_mode: Literal["chunk", "focus"] = "chunk"   # what the LLM reads: the whole retrieved chunks, or only each hit's best sentence
    bm25_norm: Literal["max", "idf"] = "idf"   # max: best chunk always scores 1.0 (junk queries too). idf: share of the query's information content matched
    max_ctx_tokens: int = 90
    queue_max: int = 8
    filler: bool = False
    speculate: bool = True           # shadow turn: quiet-host paired A/B -153 ms median, p90 1699->1348 ms (needs >=6 cores)
    speculate_min_cores: int = 6     # shadow prefill + TTS must not starve ASR on small CPUs
    yes_no_extractive: bool = True   # yes/no questions are answered by quoting the best text sentence, not the 0.5B model
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
    agc: bool = True                # gain control before ASR (quiet input: LibriSpeech WER 42.9% -> see docs/rulings.md)
    hold_ms: int = 500              # extra audio to wait when the text ends mid-sentence (0 = off)
    hold_total_cap_ms: int = 1200   # per utterance, so 'and... and...' cannot starve the reply
    endpoint_rule2_s: float = 0.4   # trailing silence after text that ends an utterance
    asr_zh_model: Literal["zipformer14m", "ctc_small"] = "ctc_small"   # AISHELL-1 CER 10.9% (wrapper, before no-reset) vs 16.0% for the 14M transducer
    llm_dir: str = "llm"   # folder under models_dir holding the .gguf (llm = Qwen2.5-0.5B, llm_q3 = Qwen3-0.6B)
    llm_temperature: float = 0.2
    llm_threads: int = 4
    max_inflight_turns: int = 2   # LLM turns running+queued; the next one gets the busy reply immediately
    access_key: str = ""            # non-empty: the first /ws message must be {"type":"auth","key":<access_key>} (a box reachable from any network). "" = off (localhost use)
    auth_timeout_s: float = 5.0     # a socket that has not authenticated by then is closed (it holds no session slot meanwhile)
    ssl_cert: str = ""
    ssl_key: str = ""
    tts_en_voice: Literal["ljspeech", "lessac"] = "ljspeech"   # ljspeech: public domain, 22.05 kHz, from scratch. lessac: Blizzard 2013 licence (custom)
    tts_en_fp32: bool = False  # int8 en voice meets the 30 MB gate but is ~3x slower on ARM/M2; fp32 is 63 MB
