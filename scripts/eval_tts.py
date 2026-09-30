"""TTS naturalness proxy: UTMOS22 (predicts MOS, trained on English speech, so zh scores are indicative only).
Usage: eval_tts.py   (downloads UTMOS22 weights via torch.hub on first run)"""
import json

import numpy as np
import torch

from vora.config import ROOT, Settings
from vora.tts import Tts

EN = ["The warranty on the VORA X200 is two years.", "Blue means the box is listening.",
      "Hold the reset button for ten seconds.", "You can import documents up to one gigabyte.",
      "Please contact support Monday to Friday.", "The speaker is ten watts.",
      "Firmware updates arrive once a month.", "It does not support Cantonese.",
      "Return shipping is paid by the buyer.", "The default wake word is Hey Vora."]
ZH = ["X200的保修期是两年。", "蓝色表示正在聆听。", "按住复位键十秒钟。", "可以导入最多一个G的文档。",
      "客服时间是周一至周五。", "扬声器是十瓦。", "固件每月更新一次。", "不支持粤语。",
      "退货运费由买家承担。", "默认唤醒词是Hey Vora。"]


def main() -> None:
    tts = Tts(Settings())
    pred = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    out = {}
    for name, sents in (("en", EN), ("zh", ZH)):
        sc = []
        for t in sents:
            pcm = np.frombuffer(b"".join(tts.synth(t)), dtype=np.int16).astype(np.float32) / 32768
            sc.append(float(pred(torch.from_numpy(pcm)[None], 16000)))
        out[name] = {"mean_mos_proxy": round(float(np.mean(sc)), 2), "min": round(min(sc), 2), "n": len(sc), "scores": [round(x, 2) for x in sc]}
        print(name, out[name]["mean_mos_proxy"], out[name]["min"])
    out["note"] = "UTMOS22 is an automatic MOS predictor trained on English; not a human MOS. zh value indicative only."
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "tts_mos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
