"""Gemini VLM engine: structured JSON output, N parallel samples, majority aggregation."""
import io
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from PIL import Image
from pydantic import BaseModel

import config


class VLMResult(BaseModel):
    defect_present: bool
    defect_type: Literal[tuple(config.DEFECT_TYPES)]
    severity: Literal["Düşük", "Orta", "Yüksek"]
    location: str
    reasoning: str
    self_confidence: float


def _jpeg(img: Image.Image, side=768) -> bytes:
    im = img.convert("RGB")
    s = side / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _empty(error, called=False, latency=0):
    return {"available": False, "called": called, "model": config.GEMINI_MODEL, "results": [],
            "majority_type": None, "consistency": 0.0, "prob": 0.0, "reasoning": "", "location": "",
            "severity": None, "latency_ms": latency, "error": error}


class VLMEngine:
    def __init__(self):
        self.key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None

    @property
    def has_key(self):
        return bool(self.key)

    def _client(self):
        if self.client is None:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=self.key,
                                       http_options=types.HttpOptions(timeout=int(config.VLM_TIMEOUT_S * 1000)))
        return self.client

    def _prompt(self, product_group, anomaly, detector):
        dets = detector.get("detections", []) if detector else []
        dtxt = "; ".join(f"{d['label_tr']} (güven {d['conf']:.2f}, kutu {[int(v) for v in d['box']]})"
                         for d in dets) or "tespit yok"
        a = anomaly or {}
        return (
            "Sen bir elektro-optik kalite muayene uzmanısın. Ürün grubu: "
            f"{product_group}. Not: görseller temsili veri (MVTec AD) örnekleridir.\n"
            f"Anomali motoru: skor {a.get('score', 0):.2f}, p-değeri {a.get('p_value', 1):.3f} "
            "(düşük p = olağandışı).\n"
            f"Nesne dedektörü kutuları: {dtxt}.\n"
            "İlk görsel orijinal, ikinci görsel (varsa) anomali ısı haritası bindirmesidir.\n"
            "Bu ipuçları yanılabilir; görselde görmediğin kusuru uydurma. Optik eksen kaçıklığı "
            "genellikle tek fotoğraftan değerlendirilemez. Kusur tipi olarak yalnızca şu listeden seç: "
            f"{', '.join(config.DEFECT_TYPES)}. Gerekçeyi Türkçe, en fazla 3 cümle yaz. "
            "self_confidence 0 ile 1 arasında olsun."
        )

    def _one(self, parts):
        from google.genai import types
        r = self._client().models.generate_content(
            model=config.GEMINI_MODEL, contents=parts,
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=VLMResult, temperature=0.4))
        if getattr(r, "parsed", None) is not None:
            res = r.parsed
        else:
            res = VLMResult.model_validate_json(r.text)
        d = res.model_dump()
        d["self_confidence"] = min(1.0, max(0.0, float(d["self_confidence"])))
        return d

    def analyze(self, image, overlay, product_group, anomaly, detector) -> dict:
        t0 = time.time()
        if not self.has_key:
            return _empty("GEMINI_API_KEY yok")
        try:
            from google.genai import types
            parts = [types.Part.from_bytes(data=_jpeg(image), mime_type="image/jpeg")]
            if overlay is not None:
                parts.append(types.Part.from_bytes(data=_jpeg(overlay), mime_type="image/jpeg"))
            parts.append(self._prompt(product_group, anomaly, detector))
        except Exception as e:
            return _empty(f"hazirlik hatasi: {e}", True)
        n = config.VLM_SAMPLES
        results, errs = [], []
        ex = ThreadPoolExecutor(max_workers=n)
        futs = [ex.submit(self._one, parts) for _ in range(n)]
        for f in futs:
            try:
                results.append(f.result(timeout=config.VLM_TIMEOUT_S))
            except Exception as e:
                errs.append(str(e)[:200])
        ex.shutdown(wait=False, cancel_futures=True)
        lat = int((time.time() - t0) * 1000)
        if not results:
            return _empty("VLM cagrilari basarisiz: " + (errs[0] if errs else "?"), True, lat)
        return aggregate(results, lat)


def aggregate(results, latency_ms=0):
    n = len(results)
    pos = [r for r in results if r["defect_present"]]
    present = len(pos) * 2 > n  # majority vote (ties -> no defect)
    if present:
        mtype = Counter(r["defect_type"] for r in pos).most_common(1)[0][0]
        agree = [r for r in results if r["defect_present"] and r["defect_type"] == mtype]
        mcount = len(pos)
    else:
        mtype = "Yok"
        agree = [r for r in results if not r["defect_present"]]
        mcount = len(agree)
    first = agree[0] if agree else results[0]
    prob = (len(pos) / n) * (sum(r["self_confidence"] for r in pos) / len(pos)) if pos else 0.0
    return {"available": True, "called": True, "model": config.GEMINI_MODEL, "results": results,
            "majority_type": mtype, "consistency": mcount / n, "prob": float(prob),
            "reasoning": first["reasoning"], "location": first["location"],
            "severity": first["severity"] if present else None, "latency_ms": latency_ms, "error": None}
