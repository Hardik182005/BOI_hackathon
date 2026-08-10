"""Evidence packet generation: JSON + printable HTML.

Every packet contains only verified pipeline outputs; synthetic/demo fields
are explicitly labelled. CSV export cells are sanitised against formula
injection.
"""
from __future__ import annotations

import datetime as dt
import html
from typing import Any


# The leading characters Excel, LibreOffice and Sheets will read as the start
# of a formula. The carriage return is easy to miss and is the one that gets
# through a naive filter: Excel strips it, then evaluates what follows.
FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value: Any) -> Any:
    """Prefix risky leading characters so spreadsheet apps treat cells as text."""
    if isinstance(value, str) and value[:1] in FORMULA_LEADS:
        return "'" + value
    return value


def build_packet(case: dict[str, Any], score: dict[str, Any],
                 narrative: dict[str, Any], actions: list[dict[str, Any]],
                 model_version: str) -> dict[str, Any]:
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "account_reference": case["account_reference"],
        "decision_tier": score["risk_tier"],
        "risk_and_uncertainty": {
            "calibrated_risk": score["calibrated_risk"],
            "model_agreement": score["model_agreement"],
            "conformal_status": score["conformal_status"],
            "ood_status": score["ood_status"],
            "anomaly_percentile": score.get("anomaly_percentile"),
        },
        "verified_reasons": score.get("top_reasons", []),
        "narrative": narrative,
        "recommended_analyst_checks": narrative.get("narrative", {}).get("recommended_checks", []),
        "analyst_action_history": actions,
        "model_version": model_version,
        "explicit_limitations": score.get("limitations", []),
    }


def packet_to_html(packet: dict[str, Any]) -> str:
    def esc(v: Any) -> str:
        return html.escape(str(v))

    reasons_rows = "".join(
        f"<tr><td>{esc(r.get('verified_semantic_name') or r['feature'])}</td>"
        f"<td>{esc(r.get('value'))}</td>"
        f"<td>{esc(r.get('legitimate_percentile'))}</td>"
        f"<td>{esc(r.get('direction'))}</td>"
        f"<td>{r.get('shap_contribution', 0):+.4f}</td></tr>"
        for r in packet["verified_reasons"]
    )
    actions_rows = "".join(
        f"<tr><td>{esc(a.get('created_utc'))}</td><td>{esc(a.get('actor'))}</td>"
        f"<td>{esc(a.get('action'))}</td><td>{esc(a.get('reason'))}</td></tr>"
        for a in packet["analyst_action_history"]
    ) or "<tr><td colspan=4>No analyst actions recorded yet</td></tr>"
    lims = "".join(f"<li>{esc(l)}</li>" for l in packet["explicit_limitations"])
    ru = packet["risk_and_uncertainty"]
    narrative = packet["narrative"].get("narrative") or {}
    return f"""<!-- printable evidence packet -->
<article style="font-family:Segoe UI,Arial,sans-serif;max-width:720px;margin:auto;color:#111">
<h1 style="border-bottom:2px solid #1e3a8a;padding-bottom:6px">MuleGuard Evidence Packet</h1>
<p><b>Account:</b> {esc(packet['account_reference'])} (masked reference)<br>
<b>Generated:</b> {esc(packet['generated_utc'])}<br>
<b>Model version:</b> {esc(packet['model_version'])}</p>
<h2>Decision</h2>
<p><b>Review tier:</b> {esc(packet['decision_tier'])} - behavioural risk assessment, not an accusation.</p>
<table border=0 cellpadding=4>
<tr><td>Calibrated risk</td><td><b>{ru['calibrated_risk']:.3f}</b></td></tr>
<tr><td>Model agreement</td><td>{ru['model_agreement']:.3f}</td></tr>
<tr><td>Conformal status</td><td>{esc(ru['conformal_status'])}</td></tr>
<tr><td>Input status</td><td>{esc(ru['ood_status'])}</td></tr>
</table>
<h2>Verified technical drivers</h2>
<table border=1 cellpadding=5 style="border-collapse:collapse;font-size:13px">
<tr><th>Feature</th><th>Value</th><th>Legit-cohort percentile</th><th>Direction</th><th>SHAP</th></tr>
{reasons_rows}
</table>
<p style="font-size:12px;color:#555">Anonymous features are compared numerically against the
legitimate cohort; no business meaning is asserted for unnamed features.</p>
<h2>Analyst summary</h2>
<p>{esc(narrative.get('summary', 'Deterministic summary unavailable'))}</p>
<p style="font-size:12px;color:#555">Narrative source: {esc(packet['narrative'].get('source'))}
{'(local LLM output, machine-validated against the scores above)' if packet['narrative'].get('source') == 'ollama' else '(deterministic template)'}</p>
<h2>Action history</h2>
<table border=1 cellpadding=5 style="border-collapse:collapse;font-size:13px">
<tr><th>When (UTC)</th><th>Actor</th><th>Action</th><th>Reason</th></tr>
{actions_rows}
</table>
<h2>Limitations</h2>
<ul>{lims}</ul>
</article>"""
