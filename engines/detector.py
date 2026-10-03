"""YOLO detector wrapper (ultralytics). Degrades to available=False if model missing."""
import time

from PIL import Image

import config


class Detector:
    def __init__(self):
        self.model = None
        self.error = None
        self.version = None
        p = config.YOLO_MODEL
        if not p.exists():
            self.error = "model yok"
            return
        try:
            from ultralytics import YOLO
            if p.suffix == ".onnx":
                task = "segment" if "seg" in p.name.lower() else "detect"
                self.model = YOLO(str(p), task=task)
            else:
                self.model = YOLO(str(p))
            self.version = p.name
        except Exception as e:
            self.error = f"model yuklenemedi: {e}"

    def predict(self, img: Image.Image) -> dict:
        t0 = time.time()
        if self.model is None:
            return {"available": False, "detections": [], "annotated": None, "latency_ms": 0, "error": self.error}
        try:
            r = self.model.predict(img.convert("RGB"), conf=config.DETECTOR_CONF, imgsz=640, verbose=False)[0]
            names = r.names
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    lab = names[int(b.cls[0])]
                    dets.append({"label": lab, "label_tr": config.map_class(lab),
                                 "conf": float(b.conf[0]), "box": [float(v) for v in b.xyxy[0].tolist()]})
            ann = Image.fromarray(r.plot()[..., ::-1].copy())
            return {"available": True, "detections": dets, "annotated": ann,
                    "latency_ms": int((time.time() - t0) * 1000), "error": None}
        except Exception as e:
            return {"available": False, "detections": [], "annotated": None,
                    "latency_ms": int((time.time() - t0) * 1000), "error": str(e)}
