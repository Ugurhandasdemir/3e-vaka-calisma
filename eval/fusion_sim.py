"""Pure offline fusion simulation reproducing fusion.py semantics for 2 engines (anomaly + detector)."""
import math
from typing import Any, Dict, Optional


def anomaly_prob(p_value: float) -> float:
    """Map a conformal p-value to a 0..1 defect probability on a log scale.
    
    Reproduces fusion.py: min(1.0, max(0.0, log(p) / log(0.02)))
    """
    p = max(float(p_value), 1e-6)
    return min(1.0, max(0.0, math.log(p) / math.log(0.02)))


def simulate_fusion(
    anomaly_p_value: float,
    detector_conf: float,
    detector_label: Optional[str] = None,
    category: str = "metal_nut",
    w_anomaly: float = 0.45,
    w_detector: float = 0.25,
    accept_below: float = 0.30,
    reject_above: float = 0.70,
    anomaly_vote_p: float = 0.053,
    detector_vote_conf: float = 0.40,
) -> Dict[str, Any]:
    """Pure offline decision fusion for Anomaly + Detector engines.
    
    Args:
        anomaly_p_value: Conformal p-value from anomaly engine.
        detector_conf: Maximum detection confidence (0.0 if no detection).
        detector_label: Top-confidence detection Turkish label (e.g. 'Yüzey çiziği') or None.
        category: Product category ('metal_nut', 'transistor', 'cable').
        w_anomaly: Anomaly engine weight.
        w_detector: Detector engine weight.
        accept_below: Defect score threshold below which ACCEPT is suggested.
        reject_above: Defect score threshold above which REJECT is suggested.
        anomaly_vote_p: p-value threshold below which anomaly votes defect (e.g. 0.053 <=> prob >= 0.75).
        detector_vote_conf: Detector confidence threshold above which detector votes defect.
    
    Returns:
        dict with keys: 'decision', 'defect_score', 'defect_type', 'confidence', 'nd', 'votes'
    """
    prob_anomaly = anomaly_prob(anomaly_p_value)
    prob_detector = float(max(0.0, min(1.0, detector_conf)))

    # Votes
    # Anomaly votes defect if p_value <= anomaly_vote_p (or prob >= threshold)
    vote_anom = anomaly_p_value <= anomaly_vote_p
    vote_det = prob_detector >= detector_vote_conf

    probs = {"anomaly": prob_anomaly, "detector": prob_detector}
    votes = {"anomaly": vote_anom, "detector": vote_det}

    wsum = w_anomaly + w_detector
    score = (w_anomaly * prob_anomaly + w_detector * prob_detector) / wsum if wsum > 0 else 0.0

    n = 2  # 2 available engines
    nd = int(vote_anom) + int(vote_det)
    agreement = max(nd, n - nd) / n

    # High risk group (Termal kamera modülü / transistor has recall history -> never auto-accept if any defect vote)
    is_high_risk = (category == "transistor" or category == "Termal kamera modülü")

    # Decision logic from fusion.py
    if score < accept_below and nd == 0:
        decision = "ACCEPT"
    elif score >= reject_above and nd >= 2:
        decision = "REJECT"
    else:
        decision = "REVIEW"

    # Risk rule: for high risk groups, any defect vote prevents ACCEPT
    if is_high_risk and nd > 0 and decision == "ACCEPT":
        decision = "REVIEW"

    # Defect type assignment
    if detector_conf > 0.0 and detector_label:
        dtype = detector_label
    else:
        dtype = "Diğer" if decision != "ACCEPT" else "Yok"

    confidence = agreement * max(score, 1.0 - score)

    return {
        "decision": decision,
        "defect_score": float(score),
        "defect_type": dtype,
        "confidence": float(confidence),
        "nd": nd,
        "votes": votes,
        "prob_anomaly": float(prob_anomaly),
        "prob_detector": float(prob_detector),
    }
