import pytest
from utils.risk_rules import combine_risk, label_from_score, RiskResult


class TestCombineRisk:
    def test_pure_text_weight(self):
        assert combine_risk(1.0, 0.0) == pytest.approx(0.6)

    def test_pure_audio_weight(self):
        assert combine_risk(0.0, 1.0) == pytest.approx(0.4)

    def test_both_high(self):
        assert combine_risk(1.0, 1.0) == pytest.approx(1.0)

    def test_both_zero(self):
        assert combine_risk(0.0, 0.0) == pytest.approx(0.0)

    def test_mixed(self):
        result = combine_risk(0.5, 0.5)
        assert result == pytest.approx(0.5)


class TestLabelFromScore:
    def test_high_label(self):
        assert label_from_score(0.9) == "high"

    def test_high_boundary(self):
        assert label_from_score(0.8) == "high"

    def test_medium_label(self):
        assert label_from_score(0.65) == "medium"

    def test_medium_boundary(self):
        assert label_from_score(0.5) == "medium"

    def test_low_label(self):
        assert label_from_score(0.3) == "low"

    def test_zero(self):
        assert label_from_score(0.0) == "low"

    def test_custom_thresholds(self):
        assert label_from_score(0.7, high=0.9, medium=0.6) == "medium"
        assert label_from_score(0.95, high=0.9, medium=0.6) == "high"
        assert label_from_score(0.4, high=0.9, medium=0.6) == "low"


class TestRiskResult:
    def test_defaults(self):
        r = RiskResult(
            text_risk=0.7,
            audio_risk=0.3,
            final_risk=0.54,
            risk_label="medium",
        )
        assert r.text == ""
        assert r.extra is None

    def test_with_extra(self):
        r = RiskResult(
            text_risk=0.9,
            audio_risk=0.8,
            final_risk=0.86,
            risk_label="high",
            text="No quiero seguir.",
            extra={"audio_features": {"rms": 0.01}},
        )
        assert r.extra["audio_features"]["rms"] == pytest.approx(0.01)
