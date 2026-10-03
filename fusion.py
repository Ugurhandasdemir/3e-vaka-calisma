"""Decision fusion (rule set).

Engine probabilities:  anomaly a = clip(log(p_value) / log(0.02), 0, 1)  (p=0.02 -> 1, p=0.05 -> 0.77,
p=0.5 -> 0.18; a plain 1 - p would give good parts ~0.5 on average because their p is uniform);  detector y = max detection conf (0 if none);
VLM v = (fraction of samples saying defect) * mean self_confidence of those.

Rules:
1. defect_score = weighted mean of probabilities over AVAILABLE engines only (config.WEIGHTS).
2. An engine votes "defect" if its probability >= config.VOTE_THRESHOLDS[engine].
3. agreement = size of the majority vote / number of available engines.
4. ACCEPT  : defect_score < ACCEPT_BELOW and every available engine votes ok (needs >= 2 engines).
   REJECT  : defect_score >= REJECT_ABOVE and >= 2 engines vote defect.
   REVIEW  : everything else (conflict, borderline score, single engine).
   A single available engine never produces an automatic ACCEPT/REJECT.
5. Early exit (pipeline): p_value > EARLY_EXIT_P and no detections -> VLM skipped, ACCEPT suggested;
   high-risk groups never take the early exit (all engines run, normal fusion).
6. Risk rule: for config.HIGH_RISK_GROUPS any defect vote means at least REVIEW (never ACCEPT).
7. confidence = agreement * max(defect_score, 1 - defect_score).
8. defect_type: VLM majority type (if not "Yok"), else top-confidence detector label, else
   "Diğer" (decision != ACCEPT) / "Yok".
"""
import math

import config


def anomaly_prob(p_value):
    """Map a conformal p-value to a 0..1 defect probability on a log scale (see module docstring)."""
    p = max(float(p_value), 1e-6)
    return min(1.0, max(0.0, math.log(p) / math.log(0.02)))


def fuse(anomaly, detector, vlm, product_group, early_exit=False):
    probs, reasons = {}, []
    if anomaly and anomaly.get("available", True) and anomaly.get("backend"):
        probs["anomaly"] = anomaly_prob(anomaly["p_value"])
    if detector and detector.get("available"):
        dets = detector.get("detections", [])
        probs["detector"] = max((d["conf"] for d in dets), default=0.0)
    if vlm and vlm.get("available"):
        probs["vlm"] = vlm.get("prob", 0.0)

    votes = {k: p >= config.VOTE_THRESHOLDS[k] for k, p in probs.items()}
    wsum = sum(config.WEIGHTS[k] for k in probs)
    score = sum(config.WEIGHTS[k] * p for k, p in probs.items()) / wsum if wsum else 0.0
    n = len(probs)
    nd = sum(votes.values())
    agreement = max(nd, n - nd) / n if n else 0.0
    high_risk = product_group in config.HIGH_RISK_GROUPS

    if "anomaly" in probs:
        reasons.append(f"Anomali: skor {anomaly['score']:.2f}, p={anomaly['p_value']:.3f} → "
                       + ("kusur oyu" if votes["anomaly"] else "sağlam oyu"))
    else:
        reasons.append("Anomali: kullanılamadı" + (f" ({anomaly.get('error')})" if anomaly and anomaly.get("error") else ""))
    if "detector" in probs:
        dets = detector.get("detections", [])
        if dets:
            reasons.append("Dedektör: " + ", ".join(f"{d['label_tr']} %{d['conf']*100:.0f}" for d in dets[:4]))
        else:
            reasons.append("Dedektör: kusur bulunamadı")
    else:
        reasons.append("Dedektör: devre dışı (model yok)" if not detector or not detector.get("error") or detector.get("error") == "model yok"
                       else f"Dedektör: kullanılamadı ({detector['error']})")
    if "vlm" in probs:
        c = round(vlm.get("consistency", 0) * config.VLM_SAMPLES)
        reasons.append(f"VLM ({c}/{config.VLM_SAMPLES} tutarlı): {vlm.get('majority_type')}")
    elif vlm and not vlm.get("called"):
        reasons.append("VLM: atlandı (erken çıkış)" if early_exit else "VLM: çağrılmadı")
    else:
        reasons.append("VLM: kullanılamadı" + (f" ({vlm.get('error')})" if vlm and vlm.get("error") else ""))

    if early_exit:
        decision = "REVIEW" if high_risk else "ACCEPT"
        reasons.append("Erken çıkış: p-değeri yüksek ve tespit yok" +
                       (" ancak termal modülde otomatik kabul yok" if high_risk else " → kabul önerisi"))
    else:
        if n >= 2 and score < config.ACCEPT_BELOW and nd == 0:
            decision = "ACCEPT"
        elif n >= 2 and score >= config.REJECT_ABOVE and nd >= 2:
            decision = "REJECT"
        else:
            decision = "REVIEW"
        if n < 2:
            reasons.append("Tek motor mevcut: otomatik karar verilmedi")
        if high_risk and nd > 0 and decision == "ACCEPT":
            decision = "REVIEW"
        if high_risk and nd > 0:
            reasons.append("Risk kuralı: termal modülde otomatik kabul yok")

    if vlm and vlm.get("available") and vlm.get("majority_type") not in (None, "Yok"):
        dtype = vlm["majority_type"]
    elif detector and detector.get("detections"):
        dtype = max(detector["detections"], key=lambda d: d["conf"])["label_tr"]
    else:
        dtype = "Diğer" if decision != "ACCEPT" else "Yok"
    severity = vlm.get("severity") if vlm and vlm.get("available") and dtype != "Yok" else None

    return {"decision": decision, "decision_label": config.DECISIONS[decision],
            "confidence": float(agreement * max(score, 1 - score)), "defect_score": float(score),
            "defect_type": dtype, "severity": severity, "reasons": reasons}
