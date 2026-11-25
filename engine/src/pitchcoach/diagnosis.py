"""Rule-based diagnosis: issue -> possible causes -> self-test -> exercises (rules_v1.yaml)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from .metrics import METRIC_DEFS
from .models import Cause, Exercise, Finding, Leak, Metric, SelfTest

_METRIC_OPS = ("z_lt", "z_gt", "abs_z_gt")
# leak.severity is |z|/3 (speed_drop) or ms/30 (out_of_order); x3 puts it on the |z| scale
_LEAK_SCALE = 3.0


def load_rules(path: str | Path | None = None) -> list[dict]:
    """Load the issue rules (the `issues` list) from YAML; packaged file when path is None."""
    if path is None:
        text = resources.files("pitchcoach").joinpath("data/rules_v1.yaml").read_text("utf-8")
    else:
        text = Path(path).read_text("utf-8")
    return list((yaml.safe_load(text) or {}).get("issues", []))


def _blank(s) -> bool:
    return not isinstance(s, str) or not s.strip()


def validate_rules(rules: list[dict]) -> list[str]:
    errors: list[str] = []
    defs = {d["name"]: d for d in METRIC_DEFS}
    for rule in rules:
        rid = rule.get("id", "?")
        for cond in (rule.get("when") or {}).get("any", []):
            if "metric" in cond:
                name = cond["metric"]
                if name not in defs:
                    errors.append(f"{rid}: unknown metric '{name}'")
                elif defs[name]["confidence"] == "low":
                    errors.append(f"{rid}: metric '{name}' has low confidence and is never "
                                  f"scored, so the condition can never trigger")
                if not any(op in cond for op in _METRIC_OPS):
                    errors.append(f"{rid}: metric condition '{name}' has no z_lt/z_gt/abs_z_gt")
            elif "leak_segment" not in cond or "leak_kind" not in cond:
                errors.append(f"{rid}: condition must have metric or leak_segment+leak_kind")
        for cause in rule.get("causes", []):
            cid = f"{rid}/{cause.get('id', '?')}"
            st = cause.get("self_test") or {}
            if len(st.get("options", [])) < 2:
                errors.append(f"{cid}: self_test needs at least 2 options")
            if _blank(cause.get("cause_source")):
                errors.append(f"{cid}: cause has empty cause_source")
            if _blank(st.get("source")):
                errors.append(f"{cid}: self_test has empty source")
            for ex in cause.get("exercises", []):
                if _blank(ex.get("source")):
                    errors.append(f"{cid}: exercise '{ex.get('id', '?')}' has empty source")
    return errors


def _match(rule: dict, by_name: dict[str, Metric], leaks: list[Leak]):
    """Return (severity, metric_hits, leak_hits) for a rule, or None if no condition holds."""
    severity, mhits, lhits = 0.0, [], []
    for cond in (rule.get("when") or {}).get("any", []):
        if "metric" in cond:
            m = by_name.get(cond["metric"])
            if m is None or m.z is None or m.value is None:
                continue
            z = m.z
            ok = ((("z_lt" in cond) and z < cond["z_lt"])
                  or (("z_gt" in cond) and z > cond["z_gt"])
                  or (("abs_z_gt" in cond) and abs(z) > cond["abs_z_gt"]))
            if ok:
                severity = max(severity, abs(z))
                mhits.append(m)
        else:
            for lk in leaks:
                if lk.segment == cond.get("leak_segment") and lk.kind == cond.get("leak_kind"):
                    severity = max(severity, lk.severity * _LEAK_SCALE)
                    lhits.append(lk)
    return (severity, mhits, lhits) if mhits or lhits else None


def _evidence(template: str, mhits: list[Metric], lhits: list[Leak]) -> str:
    parts = []
    if mhits:
        m = max(mhits, key=lambda x: abs(x.z))
        parts.append(template.replace("{value}", f"{m.value:.1f}").replace("{unit}", m.unit)
                     .replace("{z}", f"{m.z:+.1f}"))
    seen = []
    for lk in lhits:
        if lk.message not in seen:
            seen.append(lk.message)
    return " ".join(parts + seen)


def diagnose(metrics: list[Metric], leaks: list[Leak], rules: list[dict]) -> list[Finding]:
    by_name = {m.name: m for m in metrics}
    hits = []
    for rule in rules:
        res = _match(rule, by_name, leaks)
        if res is not None:
            hits.append((res[0], rule, res[1], res[2]))
    hits.sort(key=lambda h: -h[0])  # stable: ties keep file order
    out = []
    for _, rule, mhits, lhits in hits:
        causes = []
        for c in rule.get("causes", []):
            st = c["self_test"]
            causes.append(Cause(
                id=c["id"], title=c["title"],
                self_test=SelfTest(id=st["id"], instruction=st["instruction"],
                                   options=[dict(o) for o in st["options"]],
                                   implies_cause_on=st["implies_cause_on"],
                                   source=st["source"]),
                exercises=[Exercise(id=e["id"], name=e["name"], sets_reps=e["sets_reps"],
                                    description=e["description"], source=e["source"])
                           for e in c.get("exercises", [])]))
        out.append(Finding(issue_id=rule["id"], title=rule["title"],
                           evidence=_evidence(rule.get("evidence_template", ""), mhits, lhits),
                           causes=causes))
    return out
