"""Anomaly engine: EfficientAD (ONNX) if available, otherwise PatchCore (resnet18) fallback."""
import json
import os
import threading
import time

import cv2
import numpy as np
from PIL import Image

import config

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIZE = 256
BACKBONE = os.getenv("PATCHCORE_BACKBONE", "wide_resnet50_2")  # or 'resnet18'

_lock = threading.Lock()
_backbone = None          # single shared resnet18
_banks = {}               # category -> dict(bank tensor, ref float)


def _get_backbone():
    global _backbone
    with _lock:
        if _backbone is None:
            import torch
            import torchvision
            if BACKBONE == "resnet18":
                m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                m = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            _backbone = m
    return _backbone


def _prep(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB").resize((_SIZE, _SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    return a


def _features(imgs):
    """list of PIL -> tensor (N, H*W, C) of patch features (32x32 grid)."""
    import torch
    import torch.nn.functional as F
    m = _get_backbone()
    x = np.stack([(_prep(i) - _MEAN) / _STD for i in imgs]).transpose(0, 3, 1, 2)
    x = torch.from_numpy(np.ascontiguousarray(x))
    with torch.no_grad():
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x)
        l2 = m.layer2(x)
        l3 = m.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([l2, l3], 1)  # N, C, h, w
    n, c, h, w = f.shape
    return f.permute(0, 2, 3, 1).reshape(n, h * w, c), (h, w)


def _nn_dist(q, bank, chunk=1024):
    import torch
    out = []
    for i in range(0, q.shape[0], chunk):
        out.append(torch.cdist(q[i:i + chunk], bank).min(dim=1).values)
    return torch.cat(out)


def bank_file(category: str):
    return config.MODELS_DIR / f"bank_{category}_{BACKBONE}.npy"


def bank_meta_file(category: str):
    return config.MODELS_DIR / f"bank_{category}_{BACKBONE}.json"


def _build_bank(category):
    import torch
    with _lock:
        if category in _banks:
            return _banks[category]

    bf = bank_file(category)
    mf = bank_meta_file(category)
    if bf.exists():
        bank_arr = np.load(bf)
        bank = torch.from_numpy(bank_arr).float()
        ref = 1.0
        if mf.exists():
            try:
                meta = json.loads(mf.read_text())
                ref = float(meta.get("ref", 1.0))
            except Exception:
                ref = 1.0
        res = {"bank": bank, "ref": max(ref, 1e-6)}
        with _lock:
            _banks[category] = res
        return res

    files = sorted((config.SAMPLES_DIR / category / "good").glob("*.png"))
    if not files:
        raise RuntimeError(f"samples/{category}/good bos")
    feats = []
    for i in range(0, len(files), 8):
        imgs = [Image.open(p).convert("RGB") for p in files[i:i + 8]]
        f, _ = _features(imgs)
        feats.append(f.reshape(-1, f.shape[-1]))
    allf = torch.cat(feats)
    g = torch.Generator().manual_seed(0)
    k = min(allf.shape[0], max(2000, int(allf.shape[0] * 0.10)))
    idx = torch.randperm(allf.shape[0], generator=g)[:k]
    bank = allf[idx].contiguous()
    # reference scale: image-max NN distance of (a few) good images against the coreset
    maxes = []
    for f in feats[:3]:
        n = f.shape[0] // 1024
        d = _nn_dist(f, bank).reshape(n, -1)
        maxes.extend(d.max(dim=1).values.tolist())
    ref = float(np.mean(maxes)) if maxes else 1.0
    res = {"bank": bank, "ref": max(ref, 1e-6)}
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(bf, bank.cpu().numpy().astype(np.float16))
        mf.write_text(json.dumps({"ref": float(ref), "category": category, "backbone": BACKBONE}, indent=2))
    except Exception:
        pass
    with _lock:
        _banks[category] = res
    return res


class AnomalyEngine:
    def __init__(self, category: str):
        self.category = category
        self.backend = "patchcore"
        self.version = f"patchcore-{BACKBONE}"
        self._sess = None
        self._meta = {}
        onnx_path = config.anomaly_onnx(category)
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self._sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                side = onnx_path.with_suffix(".json")
                if side.exists():
                    self._meta = json.loads(side.read_text())
                self.backend = "efficientad-onnx"
                self.version = "efficientad-onnx"
            except Exception:
                self._sess = None

    # ---------------- ONNX ----------------
    def _predict_onnx(self, img):
        meta = self._meta
        inp = self._sess.get_inputs()[0]
        size = meta.get("input_size") or [_SIZE, _SIZE]
        if isinstance(size, int):
            size = [size, size]
        h, w = int(size[0]), int(size[1])
        a = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        mean = np.array(meta.get("mean", _MEAN), dtype=np.float32)
        std = np.array(meta.get("std", _STD), dtype=np.float32)
        a = (a - mean) / std
        x = a.transpose(2, 0, 1)[None].astype(np.float32)
        outs = self._sess.run(None, {meta.get("input_name", inp.name): x})
        score = None
        amap = None
        om, os_ = meta.get("map_output"), meta.get("score_output")
        names = [o.name for o in self._sess.get_outputs()]
        if om is not None or os_ is not None:
            if os_ is not None:
                score = float(np.ravel(outs[names.index(os_) if isinstance(os_, str) else os_])[0])
            if om is not None:
                amap = np.asarray(outs[names.index(om) if isinstance(om, str) else om])
        for o in outs:
            o = np.asarray(o)
            if o.size == 1 and score is None:
                score = float(o.ravel()[0])
            elif o.ndim in (3, 4) and o.size > 1 and amap is None:
                amap = o
        if amap is None:
            raise RuntimeError("onnx map output yok")
        amap = np.squeeze(amap).astype(np.float32)
        if amap.ndim != 2:
            amap = amap.reshape(amap.shape[-2:])
        if score is None:
            score = float(amap.max())
        return score, amap

    # ---------------- PatchCore ----------------
    def _predict_pc(self, img):
        import torch
        b = _build_bank(self.category)
        f, (h, w) = _features([img])
        d = _nn_dist(f[0], b["bank"]).reshape(h, w).numpy().astype(np.float32)
        return float(d.max()), d

    def predict(self, img: Image.Image) -> dict:
        t0 = time.time()
        img = img.convert("RGB")
        W, H = img.size
        err = None
        score = amap = None
        backend = self.backend
        if self._sess is not None:
            try:
                score, amap = self._predict_onnx(img)
                ref = float(self._meta.get("score_ref", 0) or 0)
            except Exception as e:  # fall back
                err = f"onnx hata: {e}"
                self._sess = None
                self.backend = backend = "patchcore"
                self.version = f"patchcore-{BACKBONE}"
                score = amap = None
        if score is None:
            try:
                score, amap = self._predict_pc(img)
                ref = _banks[self.category]["ref"]
                backend = "patchcore"
            except Exception as e:
                return {"score": 0.0, "heatmap": np.zeros((H, W), np.float32), "backend": None,
                        "latency_ms": int((time.time() - t0) * 1000), "error": f"{err or ''} {e}".strip()}
        if backend == "efficientad-onnx":
            ref = float(self._meta.get("score_ref", 0) or 0) or float(max(amap.max(), 1e-6))
        hm = cv2.resize(amap, (W, H), interpolation=cv2.INTER_CUBIC)
        hm = cv2.GaussianBlur(hm, (0, 0), 4)
        hm = np.clip((hm - 0.5 * ref) / (1.5 * ref), 0, 1).astype(np.float32)
        return {"score": float(score), "heatmap": hm, "backend": backend,
                "latency_ms": int((time.time() - t0) * 1000), "error": err}


def warm_bank(category):
    _build_bank(category)
