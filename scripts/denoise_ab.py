"""Offline A/B of the GTCRN speech enhancer in front of ASR (dev split, 60 clips per cell). Result: worse in every scene,
so it is not in the live path. Usage: python scripts/denoise_ab.py   (needs models/aux/gtcrn_simple.onnx, sha256 e77603ac...)"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import suites, eval_asr, eval_suite as E
from vora.asr import load_recognizers
from vora.config import Settings
from scripts.denoise import Denoiser
recs=load_recognizers(Settings())
noise=[suites.load_wav(suites.NOISE,r) for r in suites.load_manifest(suites.NOISE)]
def den(x):
    d=Denoiser(); out=b"".join(d.process(x[i:i+1600].tobytes()) for i in range(0,len(x),1600))+d.flush()
    return np.frombuffer(out,dtype=np.int16)
cells=[("librispeech_clean","clean"),("librispeech_clean","ambient@10"),("librispeech_clean","ambient@5"),("librispeech_clean","babble@10"),
       ("minds14_en_us","clean"),("minds14_en_us","ambient@10"),("minds14_en_us","ambient@5"),("fleurs_en","clean")]
for suite,aug in cells:
    rows=E.pick(suites.load_manifest(suite),"dev",False,60); ids=[r["id"] for r in rows]; by={r["id"]:r for r in rows}
    lp=lambda r: suites.load_wav(suite,r)
    res={}
    for mode in ("off","on"):
        hyps=[]
        for r in rows:
            x=E.apply_aug(lp(r),aug,r["id"],ids,lambda i: lp(by[i]),noise)
            if mode=="on": x=den(x)
            hyps.append(eval_asr.transcribe(recs,"en",x)[0])
        res[mode]=E.score_asr([r["text"] for r in rows],hyps,"en")["value"]
    print(f"{suite:18s} {aug:11s} off {res['off']*100:5.1f}%  on {res['on']*100:5.1f}%  delta {(res['on']-res['off'])*100:+5.1f}",flush=True)
