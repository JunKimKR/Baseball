import copy

from pitchcoach.diagnosis import diagnose, load_rules, validate_rules
from pitchcoach.metrics import METRIC_DEFS
from pitchcoach.models import Leak, Metric


def _m(name, z, value=70.0, unit="%"):
    return Metric(name=name, label=name, value=value, unit=unit, confidence="high",
                  segment="legs", score=50, z=z)


def _rule(rid, when, **extra):
    r = {
        "id": rid, "title": f"title-{rid}", "when": {"any": when},
        "evidence_template": "값 {value}{unit} (z={z})",
        "causes": [{
            "id": f"{rid}_c", "title": "원인", "cause_source": "Author et al., 2020",
            "self_test": {"id": f"{rid}_t", "instruction": "해보세요",
                          "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                          "implies_cause_on": "b", "source": "Author et al., 2020, T, J. PMID 1 DOI x"},
            "exercises": [{"id": f"{rid}_e", "name": "운동", "sets_reps": "출처의 프로토콜 참고",
                           "description": "설명", "source": "Author et al., 2020, T, J. PMID 1 DOI x"}],
        }],
    }
    r.update(extra)
    return r


RULES = [
    _rule("stride", [{"metric": "stride_length_pct_height", "z_lt": -1.0}]),
    _rule("order", [{"leak_segment": "legs", "leak_kind": "out_of_order"}]),
    _rule("gap", [{"metric": "gap_elbow_wrist_ms", "abs_z_gt": 2.0}]),
]


def test_rules_file_valid():
    rules = load_rules()
    assert len(rules) >= 6
    assert validate_rules(rules) == []


def test_diagnose_triggers_on_z():
    out = diagnose([_m("stride_length_pct_height", -1.5, 66.0)], [], RULES)
    assert [f.issue_id for f in out] == ["stride"]
    assert "66.0%" in out[0].evidence and "-1.5" in out[0].evidence
    assert out[0].causes[0].self_test.implies_cause_on == "b"
    assert out[0].causes[0].exercises[0].source
    # z_gt / abs_z_gt and non-triggering values
    assert diagnose([_m("stride_length_pct_height", -0.5)], [], RULES) == []
    assert [f.issue_id for f in diagnose([_m("gap_elbow_wrist_ms", 2.5)], [], RULES)] == ["gap"]
    assert [f.issue_id for f in diagnose([_m("gap_elbow_wrist_ms", -2.5)], [], RULES)] == ["gap"]
    z_gt = [_rule("hi", [{"metric": "stride_length_pct_height", "z_gt": 1.0}])]
    assert [f.issue_id for f in diagnose([_m("stride_length_pct_height", 1.2)], [], z_gt)] == ["hi"]


def test_diagnose_triggers_on_leak():
    leak = Leak("legs", 100, "out_of_order", 0.5, "골반 중심 최고 속도가 몸통보다 15ms 늦습니다")
    out = diagnose([], [leak], RULES)
    assert [f.issue_id for f in out] == ["order"]
    assert "15ms" in out[0].evidence
    assert diagnose([], [Leak("arm", 1, "out_of_order", 0.5, "x")], RULES) == []
    assert diagnose([], [Leak("legs", 1, "speed_drop", 0.5, "x")], RULES) == []


def test_diagnose_ignores_none_z():
    m = Metric("stride_length_pct_height", "l", 60.0, "%", "low", "legs", None, None)
    assert diagnose([m], [], RULES) == []
    assert diagnose([_m("unknown_metric", -5.0)], [], RULES) == []


def test_diagnose_sorted_by_severity():
    rules = RULES + [_rule("arm_order", [{"leak_segment": "arm", "leak_kind": "out_of_order"}])]
    metrics = [_m("stride_length_pct_height", -1.6), _m("gap_elbow_wrist_ms", 3.2)]
    leaks = [Leak("arm", 100, "out_of_order", 0.3, "m"), Leak("legs", 100, "out_of_order", 0.9, "m")]
    # leak severity x3: legs 2.7, arm 0.9 -> gap 3.2 > order 2.7 > stride 1.6 > arm_order 0.9
    out = diagnose(metrics, leaks, rules)
    assert [f.issue_id for f in out] == ["gap", "order", "stride", "arm_order"]


def test_leak_severity_on_z_scale():
    # leak severity 0.8 (= 2.4 on the z scale) outranks a metric with |z| = 1.5
    leak = Leak("legs", 100, "out_of_order", 0.8, "m")
    out = diagnose([_m("stride_length_pct_height", -1.5)], [leak], RULES)
    assert [f.issue_id for f in out] == ["order", "stride"]
    # ... but not a metric with |z| = 2.6
    out = diagnose([_m("gap_elbow_wrist_ms", 2.6)], [leak], RULES)
    assert [f.issue_id for f in out] == ["gap", "order"]


def test_diagnose_emits_self_test_source():
    out = diagnose([_m("stride_length_pct_height", -1.5)], [], RULES)
    assert out[0].causes[0].self_test.source.startswith("Author et al.")
    d = out[0].causes[0].self_test
    assert d.source and d.id == "stride_t"


def test_validate_rules_catches_missing_source():
    rules = copy.deepcopy(RULES)
    rules[0]["causes"][0]["exercises"][0]["source"] = ""
    errs = validate_rules(rules)
    assert any("source" in e for e in errs)


def test_validate_rules_catches_empty_self_test_source():
    rules = copy.deepcopy(RULES)
    rules[0]["causes"][0]["self_test"]["source"] = "  "
    assert any("self_test" in e and "source" in e for e in validate_rules(rules))


def test_validate_rules_catches_unknown_metric_and_options():
    rules = copy.deepcopy(RULES)
    rules[0]["when"]["any"][0]["metric"] = "nope"
    rules[1]["causes"][0]["self_test"]["options"] = [{"id": "a", "label": "A"}]
    errs = validate_rules(rules)
    assert any("nope" in e for e in errs)
    assert any("options" in e for e in errs)


def test_validate_rules_catches_low_confidence_metric():
    low = next(d["name"] for d in METRIC_DEFS if d["confidence"] == "low")
    rules = copy.deepcopy(RULES)
    rules[0]["when"]["any"][0]["metric"] = low
    assert any(low in e and "low" in e for e in validate_rules(rules))


def test_validate_rules_catches_empty_cause_source():
    rules = copy.deepcopy(RULES)
    rules[0]["causes"][0]["cause_source"] = ""
    assert any("cause_source" in e for e in validate_rules(rules))
