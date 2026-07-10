import { useMemo, useState } from "react";
import { api, fmtNum, fmtPct } from "../api";
import { Empty, ErrorState, HumanReviewNotice, Loading, TierBadge, usePoll } from "../components";

const TIERS = ["", "CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW", "OOD_REVIEW"];
const STATUSES = ["", "OPEN", "REVIEWED", "CLOSED"];

export default function AlertQueue() {
  const [tier, setTier] = useState("");
  const [status, setStatus] = useState("OPEN");
  const [sortKey, setSortKey] = useState<"calibrated_risk" | "created_utc">("calibrated_risk");

  const params = useMemo(() => {
    const q = new URLSearchParams();
    if (tier) q.set("tier", tier);
    if (status) q.set("status", status);
    q.set("limit", "500");
    return `?${q.toString()}`;
  }, [tier, status]);

  const { data, error, loading } = usePoll(() => api.cases(params), [params], 10000);

  const rows = useMemo(() => {
    const list: any[] = data?.cases ?? [];
    return [...list].sort((a, b) =>
      sortKey === "calibrated_risk"
        ? b.calibrated_risk - a.calibrated_risk
        : String(b.created_utc).localeCompare(String(a.created_utc))
    );
  }, [data, sortKey]);

  return (
    <>
      <h2 className="page-title">Alert Queue</h2>
      <p className="page-sub">
        Ranked review queue. Every entry is a recommendation for human review —
        never an automatic action.
      </p>
      <HumanReviewNotice />
      <div className="card" style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <label>Tier{" "}
          <select value={tier} onChange={(e) => setTier(e.target.value)}>
            {TIERS.map((t) => <option key={t} value={t}>{t || "all"}</option>)}
          </select>
        </label>
        <label>Status{" "}
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            {STATUSES.map((s) => <option key={s} value={s}>{s || "all"}</option>)}
          </select>
        </label>
        <label>Sort by{" "}
          <select value={sortKey} onChange={(e) => setSortKey(e.target.value as any)}>
            <option value="calibrated_risk">calibrated risk</option>
            <option value="created_utc">newest first</option>
          </select>
        </label>
      </div>
      {loading ? <Loading what="alert queue" /> :
       error ? <ErrorState msg={error} /> :
       rows.length === 0 ? (
        <Empty msg="No cases match the filters. Score accounts through the API or run the demo script to create review cases." />
      ) : (
        <div className="card" style={{ marginTop: 14, overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Case</th><th>Account (masked)</th><th>Tier</th>
                <th>Calibrated risk</th><th>Status</th><th>Assignee</th><th>Created (UTC)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.case_id}>
                  <td><a href={`#/cases/${c.case_id}`}>{c.case_id}</a></td>
                  <td>{c.account_reference}</td>
                  <td><TierBadge tier={c.risk_tier} /></td>
                  <td><b>{fmtNum(c.calibrated_risk)}</b></td>
                  <td>{c.status}</td>
                  <td>{c.assignee ?? "—"}</td>
                  <td style={{ fontSize: 12, color: "var(--muted)" }}>{String(c.created_utc).slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
