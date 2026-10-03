# Interface contract (all lanes)

Constants: `config.py` (owned by Opus — do not edit; ask if something is missing).
Full plan: `docs/PLAN.md` (Turkish). Architecture notes: `docs/MIMARI.md`.

## pipeline.run(image: PIL.Image.Image, product_group: str) -> dict

`product_group` is a key of `config.PRODUCT_GROUPS`. Never raises for engine failures; a failed or
missing engine has `available=False` and an `error` string.

```python
{
  "decision": "ACCEPT" | "REVIEW" | "REJECT",
  "decision_label": str,            # config.DECISIONS[decision]
  "confidence": float,              # 0..1, shown as %
  "defect_score": float,            # 0..1 fused
  "defect_type": str,               # one of config.DEFECT_TYPES
  "severity": str | None,           # "Düşük" | "Orta" | "Yüksek" | None
  "reasons": list[str],             # Turkish, human readable, one per engine + rule hits
  "heatmap_overlay": PIL.Image | None,
  "detection_image": PIL.Image | None,
  "engines": {
    "anomaly":  {"available": bool, "backend": "efficientad-onnx" | "patchcore" | None,
                 "score": float, "p_value": float, "prob": float, "latency_ms": int, "error": str | None},
    "detector": {"available": bool, "detections": [{"label": str, "label_tr": str, "conf": float,
                 "box": [x1, y1, x2, y2]}], "prob": float, "latency_ms": int, "error": str | None},
    "vlm":      {"available": bool, "called": bool, "model": str, "results": list[dict],
                 "majority_type": str | None, "consistency": float, "prob": float,
                 "reasoning": str, "location": str, "latency_ms": int, "error": str | None},
  },
  "model_versions": dict,           # e.g. {"anomaly": "...", "detector": "...", "vlm": "...", "app": config.APP_VERSION}
  "latency_ms": int,
}
```

## storage (Lane C) uses only the dict above plus UI fields:
inspector, serial_no, human_decision ("ACCEPT"/"REJECT"), human_defect_type, note.
