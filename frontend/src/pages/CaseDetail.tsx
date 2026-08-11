import { useState } from "react";
import { useParams } from "react-router-dom";
import { api, fmtNum, fmtPct } from "../api";
import { Empty, ErrorState, HumanReviewNotice, Loading, TierBadge, usePoll } from "../components";

export default function CaseDetail() {
  const { caseId = "" } = useParams();
  const [refresh, setRefresh] = useState(0);
  const { data, error, loading } = usePoll(() => api.caseDetail(caseId), [caseId, refresh]);
  const [actor, setActor] = useState("analyst.demo");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<any>(null);
  const [reportErr, setReportErr] = useState<string | null>(null);

  if (loading) return <Loading what={`case ${caseId}`} />;
  if (error) return <ErrorState msg={error} />;
  if (!data) return <Empty msg="Case not found." />;

  const { case: c, score, actions, feedback } = data as any;
  const reasons: any[] = score?.top_reasons ?? [];

  const act = async (action: string, approved_by?: string) => {
    if (!reason.trim() || busy) return;
    setBusy(true);
    try {
      await api.decision(caseId, { actor, action, reason, approved_by });
      setReason("");
      setRefresh((r) => r + 1);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setBusy(false);
    }
  };

  const genReport = async (useLlm: boolean) => {
    setBusy(true);
    setReportErr(null);
    try {
      setReport(await api.generateReport(caseId, useLlm));
    } catch (e: any) {
      setReportErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h2 className="page-title">Case {caseId}</h2>
      <p className="page-sub">
        Account {c.account_reference} (masked reference) · status {c.status} ·{" "}
        <a href={`#/proof/${caseId}`}>open the dual-evidence ProofGraph</a>
      </p>
      <HumanReviewNotice />

      <div className="grid cols-4">
        <div className="card">
          <h3>Decision tier</h3>
          <div className="stat"><TierBadge tier={c.risk_tier} /></div>
          <div className="stat-sub">behavioural risk — not an accusation</div>
        </div>
        <div className="card">
          <h3>Calibrated risk</h3>
          <div className="stat">{fmtNum(c.calibrated_risk)}</div>
          <div className="stat-sub">model agreement {fmtNum(score?.model_agreement)}</div>
        </div>
        <div className="card">
          <h3>Uncertainty</h3>
          <div className="stat" style={{ fontSize: 16 }}>{score?.conformal_status ?? "–"}</div>
          <div className="stat-sub">conformal prediction set (dev-calibrated)</div>
        </div>
        <div className="card">
          <h3>Input status</h3>
          <div className="stat" style={{ fontSize: 16 }}>{score?.ood_status ?? "–"}</div>
          <div className="stat-sub">anomaly percentile {fmtNum(score?.anomaly_percentile, 1)}</div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Verified technical drivers (SHAP, vs legitimate cohort)</h3>
          {reasons.length ? (
            <table className="data">
              <thead>
                <tr><th>Feature</th><th>Value</th><th>Legit percentile</th><th>Effect</th><th>SHAP</th></tr>
              </thead>
              <tbody>
                {reasons.map((r, i) => (
                  <tr key={i}>
                    <td>{r.verified_semantic_name ?? r.feature}</td>
                    <td>{r.value == null ? "missing" : fmtNum(r.value, 3)}</td>
                    <td>{r.legitimate_percentile == null ? "–" : `${r.legitimate_percentile.toFixed(1)}%`}</td>
                    <td>
                      <span className={`badge ${r.direction === "INCREASES_RISK" ? "danger" : "ok"}`}>
                        {r.direction === "INCREASES_RISK" ? "↑ risk" : "↓ risk"}
                      </span>
                    </td>
                    <td>{r.shap_contribution >= 0 ? "+" : ""}{fmtNum(r.shap_contribution, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty msg="Explanations not stored for this score." />}
          <div className="notice">
            Anonymous features are compared numerically against the legitimate
            cohort. No business meaning is asserted for unnamed features.
          </div>
        </div>

        <div className="card">
          <h3>Model comparison (raw scores)</h3>
          {score?.raw_scores ? (
            <table className="data">
              <tbody>
                {Object.entries(score.raw_scores).map(([m, v]: any) => (
                  <tr key={m}><td>{m}</td><td><b>{fmtNum(v)}</b></td></tr>
                ))}
                <tr><td>hard-negative verifier</td>
                  <td>{score.verifier_confirms_risk
                    ? <span className="badge danger">confirms mule-like pattern</span>
                    : <span className="badge ok">look-alike pattern — protection engaged</span>}
                  </td></tr>
              </tbody>
            </table>
          ) : <Empty msg="No stored score payload." />}
          <h3 style={{ marginTop: 16 }}>Policy reasons</h3>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
            {(score?.reasons ?? []).map((r: string, i: number) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Analyst actions</h3>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
            <input placeholder="your analyst id" value={actor}
                   onChange={(e) => setActor(e.target.value)} style={{ width: 140 }} />
            <input placeholder="reason (required)" value={reason}
                   onChange={(e) => setReason(e.target.value)} style={{ flex: 1, minWidth: 180 }} />
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn" disabled={busy || !reason.trim()} onClick={() => act("ASSIGN")}>Assign to me</button>
            <button className="btn secondary" disabled={busy || !reason.trim()} onClick={() => act("MARK_REVIEWED")}>Mark reviewed</button>
            <button className="btn secondary" disabled={busy || !reason.trim()} onClick={() => act("ESCALATE")}>Escalate</button>
            <button className="btn secondary" disabled={busy || !reason.trim()}
                    onClick={() => {
                      const approver = window.prompt("RECOMMEND FREEZE requires a second authorised approver id:");
                      if (approver) act("RECOMMEND_FREEZE", approver);
                    }}>
              Recommend freeze (needs approver)
            </button>
            <button className="btn secondary" disabled={busy || !reason.trim()} onClick={() => act("CLOSE")}>Close</button>
          </div>
          <table className="data" style={{ marginTop: 12 }}>
            <thead><tr><th>When</th><th>Actor</th><th>Action</th><th>Reason</th></tr></thead>
            <tbody>
              {actions.length ? actions.map((a: any) => (
                <tr key={a.id}>
                  <td style={{ fontSize: 12 }}>{String(a.created_utc).slice(0, 19)}</td>
                  <td>{a.actor}</td>
                  <td>{a.action}{a.approved_by ? ` (approved: ${a.approved_by})` : ""}</td>
                  <td>{a.reason}</td>
                </tr>
              )) : <tr><td colSpan={4} style={{ color: "var(--muted)" }}>No actions yet</td></tr>}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Evidence report</h3>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn" disabled={busy} onClick={() => genReport(true)}>
              Generate (local LLM if available)
            </button>
            <button className="btn secondary" disabled={busy} onClick={() => genReport(false)}>
              Generate (deterministic)
            </button>
          </div>
          {reportErr && <ErrorState msg={reportErr} />}
          {report && (
            <div style={{ marginTop: 12 }}>
              <div>
                <span className={`badge ${report.narrative.source === "ollama" ? "info" : "ok"}`}>
                  narrative source: {report.narrative.source}
                </span>
                {report.narrative.llm_rejected_reasons?.length > 0 && (
                  <span className="badge warn" style={{ marginLeft: 8 }}>
                    LLM output rejected: {report.narrative.llm_rejected_reasons.join("; ")}
                  </span>
                )}
              </div>
              <p style={{ fontSize: 13.5 }}>{report.narrative.narrative?.summary}</p>
              <b style={{ fontSize: 12 }}>Recommended checks</b>
              <ul style={{ marginTop: 4, fontSize: 13 }}>
                {(report.narrative.narrative?.recommended_checks ?? []).map((c: string) => <li key={c}>{c}</li>)}
              </ul>
              <b style={{ fontSize: 12 }}>Limitations</b>
              <ul style={{ marginTop: 4, fontSize: 12.5, color: "var(--muted)" }}>
                {(report.narrative.narrative?.limitations ?? []).map((l: string) => <li key={l}>{l}</li>)}
              </ul>
              <button className="btn secondary"
                      onClick={() => {
                        const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
                        const a = document.createElement("a");
                        a.href = URL.createObjectURL(blob);
                        a.download = `${caseId}_evidence.json`;
                        a.click();
                      }}>
                Export evidence packet (JSON)
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Analyst feedback (feeds retraining queue)</h3>
        <FeedbackForm caseId={caseId} onDone={() => setRefresh((r) => r + 1)} />
        <table className="data" style={{ marginTop: 10 }}>
          <thead><tr><th>When</th><th>Actor</th><th>Verdict</th><th>Notes</th></tr></thead>
          <tbody>
            {feedback.length ? feedback.map((f: any) => (
              <tr key={f.id}>
                <td style={{ fontSize: 12 }}>{String(f.created_utc).slice(0, 19)}</td>
                <td>{f.actor}</td><td>{f.verdict}</td><td>{f.notes ?? "—"}</td>
              </tr>
            )) : <tr><td colSpan={4} style={{ color: "var(--muted)" }}>No feedback yet</td></tr>}
          </tbody>
        </table>
      </div>
    </>
  );
}

function FeedbackForm({ caseId, onDone }: { caseId: string; onDone: () => void }) {
  const [actor, setActor] = useState("analyst.demo");
  const [verdict, setVerdict] = useState("INCONCLUSIVE");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      <input value={actor} onChange={(e) => setActor(e.target.value)} style={{ width: 140 }} />
      <select value={verdict} onChange={(e) => setVerdict(e.target.value)}>
        <option>CONFIRMED_MULE</option>
        <option>FALSE_POSITIVE</option>
        <option>INCONCLUSIVE</option>
      </select>
      <input placeholder="notes" value={notes} onChange={(e) => setNotes(e.target.value)}
             style={{ flex: 1, minWidth: 160 }} />
      <button className="btn" disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await api.feedback(caseId, { actor, verdict, notes: notes || undefined });
                  setNotes("");
                  onDone();
                } catch (e: any) {
                  alert(e.message);
                } finally {
                  setBusy(false);
                }
              }}>
        Submit verdict
      </button>
    </div>
  );
}
