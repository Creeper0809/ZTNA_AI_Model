from ztna_ueba.policy import map_risk_to_policy


def test_five_stage_policy_boundaries():
    assert map_risk_to_policy(0.19, False)["stage"] == "allow"
    assert map_risk_to_policy(0.20, False)["stage"] == "monitor"
    assert map_risk_to_policy(0.50, False)["stage"] == "step_up"
    assert map_risk_to_policy(0.75, False)["stage"] == "restrict"
    assert map_risk_to_policy(0.90, False)["stage"] == "deny"


def test_shadow_mode_suppresses_enforcement_at_any_risk():
    decision = map_risk_to_policy(1.0, True)
    assert decision["stage"] == "shadow"
    assert decision["enforcement_allowed"] is False
