"""Paired A/B (bench --speculate ab): per-question end-to-end totals for arm off vs on, answered en turns only."""
import json
import sys

import numpy as np


def main(path: str) -> None:
    r = json.load(open(path))
    by = {}
    for x in r["rows"]:
        if x.get("context") and "total" in x and x.get("lang") == "en":
            by.setdefault(x["q"], {})[x["arm"]] = x
    pairs = [(v["off"]["total"], v["on"]["total"]) for v in by.values() if "off" in v and "on" in v]
    off, on = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    d = on - off
    hit = sum(bool(v["on"].get("speculated")) for v in by.values() if "on" in v)
    print(f"pairs {len(pairs)} (en, answered) | OFF p50 {np.percentile(off,50):.0f} p90 {np.percentile(off,90):.0f} | "
          f"ON p50 {np.percentile(on,50):.0f} p90 {np.percentile(on,90):.0f} | paired delta median {np.median(d):.0f} ms "
          f"(neg = faster), ON faster in {int((d < 0).sum())}/{len(d)} | shadow promoted {hit} | host load {r['host'].get('loadavg_1m')}")


if __name__ == "__main__":
    main(sys.argv[1])
