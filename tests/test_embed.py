import numpy as np
import pytest

from vora.config import Settings
from vora.rag.embed import Embedder, model_dir

S = Settings()
pytestmark = pytest.mark.skipif(not (model_dir(S, "zh") / "tokenizer.json").exists(), reason="python scripts/fetch_models.py (embed_zh / embed_en)")


@pytest.fixture(scope="module", params=["en", "zh"])
def emb(request):
    return request.param, Embedder(model_dir(S, request.param))


def test_unit_norm_and_shape(emb):
    _, e = emb
    v = e.embed(["warranty of the X200", "保修期多久"])
    assert v.shape[0] == 2 and v.dtype == np.float32 and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)


def test_empty_string_ok_and_long_text_truncated(emb):
    _, e = emb
    v = e.embed(["", "word " * 2000])
    assert np.isfinite(v).all() and v.shape[0] == 2


def test_batch_equals_single(emb):     # int8 activations are quantized per tensor: the embedder runs one text at a time
    _, e = emb
    texts = ["how long is the warranty", "what does the red light mean", "能说英语吗"]
    b = e.embed(texts)
    s = np.vstack([e.embed([t]) for t in texts])
    assert np.allclose(b, s, atol=2e-3)                      # padding must not change a sentence's vector


def test_int8_matches_fp32_reference(emb):
    """Dev-only oracle: the fp32 fastembed model of the same checkpoint (skipped when fastembed is not installed).
    int8 lands at cosine 0.96-0.99; what matters was checked at retrieval level (docs/rulings.md: same top-3 on the demo
    sets, bank zh dev 97% -> 94%)."""
    fastembed = pytest.importorskip("fastembed")
    lang, e = emb
    name = {"zh": "BAAI/bge-small-zh-v1.5", "en": "BAAI/bge-small-en-v1.5"}[lang]
    ref = fastembed.TextEmbedding(name, threads=1)
    texts = ["the X200 warranty period is 2 years", "hold the reset button for 10 seconds", "X200 的保修期是 2 年", "Hey Vora 唤醒词"]
    a = e.embed(texts)
    r = np.array(list(ref.embed(texts)), dtype=np.float32)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    assert (a * r).sum(1).min() >= 0.95
