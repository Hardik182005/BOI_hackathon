import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, fmtNum, fmtPct } from "./api";
import { TierBadge } from "./components";

/** The judge-facing "can I test this myself?" entry point.
 *
 * Two paths are offered and both run the real scorer. There is deliberately no
 * short "enter 8 friendly fields" form: the serving contract requires every one
 * of the model's selected features, and a missing one is answered with 422
 * SCHEMA_ERROR rather than a silent zero-fill. A form that collected a handful
 * of human-readable fields could therefore only produce a synthetic number, and
 * printing that next to a real risk score - in the same panel, in the same font
 * - is the one thing this dashboard is built not to do. The constraint is
 * stated in the panel instead of being worked around, because "we refuse to
 * guess the other 112 columns" is the feature, not a limitation to hide.
 */

type Props = { cases: any[] };

export default function TryPanel({ cases }: Props) {
  const nav = useNavigate();
  const [open, setOpen] = useState(false);
  const [ref, setRef] = useState("DEMO-JUDGE-1");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  const scored = cases.filter((c) => c.case_id);

  const submit = async () => {
    if (busy) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    let features: any;
    try {
      features = JSON.parse(text);
    } catch {
      setErr("That is not valid JSON. Paste an object of raw feature name -> value.");
      setBusy(false);
      return;
    }
    if (features && typeof features === "object" && features.features) {
      features = features.features; // accept a full request body too
    }
    try {
      setResult(await api.score({ account_reference: ref, features }));
    } catch (e: any) {
      setErr(e?.message ?? "scoring failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="card try-panel">
        <div className="try-head">
          <div>
            <h3 style={{ marginBottom: 4 }}>Try MuleGuard</h3>
            <div className="stat-sub">
              Score an individual account, or upload an unseen validation
              dataset. The model never retrains on uploaded validation data.
            </div>
          </div>
          <div className="try-actions">
            <button className="btn" onClick={() => setOpen(true)}>
              Score Single Account
            </button>
            <button className="btn secondary" onClick={() => nav("/validation")}>
              Upload Validation Dataset
            </button>
          </div>
        </div>
      </div>

      {open ? (
        <div
          className="modal-backdrop"
          onClick={(e) => {
            if (e.target === e.currentTarget) setOpen(false);
          }}
        >
          <div className="modal" role="dialog" aria-label="Score a single account">
            <div className="modal-head">
              <h3>Score a single account</h3>
              <button className="btn secondary" onClick={() => setOpen(false)}>
                Close
              </button>
            </div>

            <div className="notice human" style={{ marginTop: 0 }}>
              <b>Both paths below run the deployed scorer.</b> There is no
              short-form "enter a few fields" demo: the model requires all of its
              selected features, and a missing one returns{" "}
              <code>422 SCHEMA_ERROR</code> instead of being quietly filled with
              zero. A synthetic number is not shown beside a real one.
            </div>

            <h4 className="modal-sub">1 · Open an account that has already been scored</h4>
            <p className="stat-sub">
              Full evidence for a real scored account: calibrated risk, tier,
              model consensus, SHAP reason codes, false-positive context and the
              ProofGraph.
            </p>
            {scored.length ? (
              <table className="data">
                <thead>
                  <tr><th>Case</th><th>Tier</th><th>Risk</th><th /></tr>
                </thead>
                <tbody>
                  {scored.slice(0, 6).map((c) => (
                    <tr key={c.case_id}>
                      <td><code>{c.account_reference ?? c.case_id}</code></td>
                      <td><TierBadge tier={c.risk_tier} /></td>
                      <td>{fmtNum(c.calibrated_risk)}</td>
                      <td>
                        <button
                          className="btn secondary"
                          onClick={() => { setOpen(false); nav(`/cases/${c.case_id}`); }}
                        >
                          Open
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty">
                No scored cases yet — run the demo script or score an account
                below to populate the queue.
              </div>
            )}

            <h4 className="modal-sub">2 · Score a raw account now</h4>
            <p className="stat-sub">
              Paste a raw feature object (<code>{"{ \"F0001\": 12.5, … }"}</code>)
              — a single row of the dataset in JSON. This is sent to{" "}
              <code>POST /v1/score</code> and scored by the same bundle that
              serves every other number on this dashboard.
            </p>
            <div className="try-form">
              <label>
                Account reference <span className="stat-sub">(masked or synthetic id)</span>
                <input value={ref} onChange={(e) => setRef(e.target.value)} maxLength={64} />
              </label>
              <label>
                Raw features (JSON)
                <textarea
                  rows={6}
                  value={text}
                  placeholder='{"F0001": 12.5, "F0002": 0, ...}'
                  onChange={(e) => setText(e.target.value)}
                />
              </label>
              <button className="btn" onClick={submit} disabled={busy || !text.trim()}>
                {busy ? "Scoring…" : "Score account"}
              </button>
            </div>

            {err ? (
              <div className="error-state" style={{ marginTop: 12 }}>
                <b>Not scored.</b>
                <div style={{ marginTop: 6 }}>{err}</div>
                <div style={{ marginTop: 6, fontSize: 12 }}>
                  A refusal here is the firewall working: the scorer will not
                  invent a value for a feature you did not supply.
                </div>
              </div>
            ) : null}

            {result ? (
              <div className="card" style={{ marginTop: 12 }}>
                <h3>Result — deployed scorer</h3>
                <div className="try-result">
                  <div>
                    <div className="stat">{fmtPct(result.calibrated_risk, 1)}</div>
                    <div className="stat-sub">calibrated mule risk</div>
                  </div>
                  <div>
                    <div style={{ marginBottom: 6 }}>
                      <TierBadge tier={result.risk_tier ?? "MONITOR"} />
                    </div>
                    <div className="stat-sub">
                      review tier — a recommendation to an analyst, never an
                      automatic action
                    </div>
                  </div>
                </div>
                <table className="data" style={{ marginTop: 10 }}>
                  <tbody>
                    <tr>
                      <td>Model consensus</td>
                      <td>
                        agreement {fmtNum(result.model_agreement)} · spread{" "}
                        {fmtNum(result.prediction_std)}
                      </td>
                    </tr>
                    <tr>
                      <td>Conformal / OOD</td>
                      <td>
                        <b>{result.conformal_status ?? "—"}</b> ·{" "}
                        <b>{result.ood_status ?? "—"}</b>
                      </td>
                    </tr>
                    <tr>
                      <td>Look-alike verifier</td>
                      <td>
                        {result.verifier_confirms_risk ? "confirms risk" : "does not confirm risk"}
                        {" "}({fmtNum(result.verifier_probability)})
                      </td>
                    </tr>
                    <tr>
                      <td>Decision</td>
                      <td>
                        <b>{result.decision ?? "—"}</b>
                        {result.auto_action ? null : " · no automatic action"}
                      </td>
                    </tr>
                  </tbody>
                </table>

                {Array.isArray(result.reasons) && result.reasons.length ? (
                  <>
                    <h4 className="modal-sub">Why this tier</h4>
                    <ul className="reason-list">
                      {result.reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}
                    </ul>
                  </>
                ) : null}

                {Array.isArray(result.top_reasons) && result.top_reasons.length ? (
                  <>
                    <h4 className="modal-sub">
                      Top drivers — with legitimate-cohort context
                    </h4>
                    <table className="data">
                      <thead>
                        <tr>
                          <th>Feature</th><th>Value</th>
                          <th>Legit. median</th><th>Pctile</th><th>SHAP</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.top_reasons.slice(0, 5).map((r: any, i: number) => (
                          <tr key={i}>
                            <td><code>{r.verified_semantic_name ?? r.feature}</code></td>
                            <td>{fmtNum(r.value, 2)}</td>
                            <td>{fmtNum(r.legitimate_cohort_median, 2)}</td>
                            <td>{fmtNum(r.legitimate_percentile, 1)}</td>
                            <td>{fmtNum(r.shap_contribution)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="stat-sub" style={{ marginTop: 6 }}>
                      "Legit. median" and "Pctile" place this account against the
                      legitimate cohort — the false-positive context that keeps a
                      high value from reading as guilt on its own.
                    </div>
                  </>
                ) : null}

                {result.case_id ? (
                  <button
                    className="btn secondary"
                    style={{ marginTop: 12 }}
                    onClick={() => { setOpen(false); nav(`/cases/${result.case_id}`); }}
                  >
                    Open full case · SHAP, ProofGraph, analyst actions
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </>
  );
}
