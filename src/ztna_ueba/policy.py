"""Five-stage ZTNA policy mapping for calibrated risk scores."""

from __future__ import annotations


def map_risk_to_policy(risk_score: float, shadow_mode_required: bool) -> dict:
    risk = max(0.0, min(1.0, float(risk_score)))
    if shadow_mode_required:
        return {
            "stage": "shadow",
            "action": "observe_only",
            "enforcement_allowed": False,
            "reason": "baseline, calibration, or OOD validation is incomplete",
        }
    if risk < 0.20:
        stage, action = "allow", "allow"
    elif risk < 0.50:
        stage, action = "monitor", "allow_and_monitor"
    elif risk < 0.75:
        stage, action = "step_up", "require_mfa_or_reauthentication"
    elif risk < 0.90:
        stage, action = "restrict", "least_privilege_restriction"
    else:
        stage, action = "deny", "deny_or_isolate"
    return {
        "stage": stage,
        "action": action,
        "enforcement_allowed": True,
        "reason": "calibrated risk policy",
    }
