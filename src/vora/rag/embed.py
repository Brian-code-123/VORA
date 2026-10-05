"""Sentence embeddings with int8 ONNX models (Xenova exports of BAAI bge-small zh / en) run directly on onnxruntime.
fastembed loaded the fp32 models with onnxruntime's memory arena: ~156 MB USS for both languages on Linux arm64 (G3 is
500 MB for ASR+RAG+TTS). CLS pooling + L2 normalisation, as bge expects."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from vora.config import Settings

MAX_TOKENS = 512


def model_dir(settings: Settings, lang: str) -> Path:
    return settings.models_dir / f"embed_{lang}"


class Embedder:
    def __init__(self, d: Path, threads: int = 1):
        onnx = next((p for p in (d / "onnx" / "model_int8.onnx", d / "model_int8.onnx") if p.exists()), None)
        if onnx is None:
            raise FileNotFoundError(f"no int8 embedding model in {d}: python scripts/fetch_models.py")
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False          # the arena kept tens of MB per session for one-sentence batches
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)

    def embed(self, texts: list[str]) -> np.ndarray:
        """One text per run: the int8 graph quantizes activations per tensor, so padding in a batch shifts the vectors."""
        return np.vstack([self._one(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def _one(self, text: str) -> np.ndarray:
        e = self.tok.encode(text)
        ids = np.array([e.ids], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
        if "token_type_ids" in self.names:
            feed["token_type_ids"] = np.zeros_like(ids)
        cls = self.sess.run(None, feed)[0][:, 0].astype(np.float32)
        return cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-9)
