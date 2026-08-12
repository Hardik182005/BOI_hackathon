import { api, fmtNum, fmtPct } from "../api";
import { Empty, ErrorState, HumanReviewNotice, Loading, TierBadge, usePoll } from "../components";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const TIER_ORDER = ["CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW", "OOD_REVIEW", "MONITOR"];
const TIER_COLORS: Record<string, string> = {
  CRITICAL_REVIEW: "#ef4444", URGENT_REVIEW: "#f59e0b",
  STANDARD_REVIEW: "#3b82f6", OOD_REVIEW: "#8b5cf6", MONITOR: "#10b981",
};

export default function Overview() {
  const metrics = usePoll(() => api.metrics(), []);
  const health = usePoll(() => api.health(), [], 15000);
  const cases = usePoll(() => api.cases("?limit=500"), [], 15000);
  const drift = usePoll(() => api.drift(), []);

  if (metrics.loading) return <Loading what="executive overview" />;
  if (metrics.error) return <ErrorState msg={metrics.error} />;

  const lt = metrics.data?.locked_test;
  const lens = metrics.data?.lens_stack_oof;
  const tierRows: any[] = lt?.tier_distribution ?? [];
  const tierData = TIER_ORDER.map((t) => ({
    tier: t.replace("_REVIEW", "").replace("MONITOR", "MONITOR"),
    full: t,
    n: tierRows.find((r) => r.tier === t)?.n ?? 0,
  }));
  const caseList: any[] = cases.data?.cases ?? [];
  const openCases = caseList.filter((c) => c.status === "OPEN");
  const budget = lt?.recall_at_budget ?? [];

  return (
    <>
      <h2 className="page-title">Executive Overview</h2>
      <p className="page-sub">
        All figures come from pipeline artifacts (locked-test evaluation +
        dev out-of-fold). Split and metric are stated on every panel.
      </p>
      <HumanReviewNotice />
      <div className="grid cols-4">
        <div className="card">
          <h3>Accounts scored (locked test)</h3>
          <div className="stat">{lt ? lt.n.toLocaleString() : "–"}</div>
          <div className="stat-sub">true mules present: {lt?.n_positives ?? "–"} ({fmtPct(lt?.prevalence, 2)})</div>
        </div>
        <div
          className="card"
          title={
            "PR-AUC and recall at analyst budget are the primary performance " +
            "measures. Plain accuracy is misleading under the ~1% mule " +
            "prevalence and is deliberately not shown: a model that flags " +
            "nothing would score ~99% while catching zero mules. ROC-AUC is " +
            "context only."
          }
        >
          <h3>PR-AUC · locked test <span className="stat-sub">(primary metric)</span></h3>
          <div className="stat">{fmtNum(lt?.pr_auc?.point)}</div>
          <div className="stat-sub">
            95% CI {fmtNum(lt?.pr_auc?.ci_low)}–{fmtNum(lt?.pr_auc?.ci_high)} ·
            no-skill = {fmtNum(lt?.prevalence, 4)}
          </div>
        </div>
        <div className="card">
          <h3>Calibration health</h3>
          <div className="stat">{lt?.brier != null ? lt.brier.toFixed(5) : "–"}</div>
          <div className="stat-sub">
            Brier score · ECE {fmtNum(lt?.calibration?.ece, 4)} (locked test)
          </div>
        </div>
        <div className="card">
          <h3>Drift status</h3>
          <div className="stat">
            <span className={`badge ${drift.data?.latest?.status === "STABLE" ? "ok" : drift.data?.latest?.status === "MODERATE" ? "warn" : drift.data?.latest ? "danger" : "info"}`}>
              {drift.data?.latest?.status ?? "no snapshot"}
            </span>
          </div>
          <div className="stat-sub">score PSI {fmtNum(drift.data?.latest?.score_psi, 4)}</div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Risk-tier distribution — locked test (policy applied)</h3>
          {lt ? (
            <div className="chart-box">
              <ResponsiveContainer>
                <BarChart data={tierData} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
                  <XAxis dataKey="tier" stroke="#5b6675" fontSize={11} />
                  <YAxis stroke="#5b6675" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: "#ffffff", border: "1px solid #dde3ec", color: "#111" }}
                    formatter={(v: any) => [v, "accounts"]}
                  />
                  <Bar dataKey="n" radius={[4, 4, 0, 0]}>
                    {tierData.map((d) => (
                      <Cell key={d.full} fill={TIER_COLORS[d.full]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Empty msg="Locked-test evaluation not run yet." />
          )}
        </div>
        <div className="card">
          <h3>Recall at analyst alert budgets — locked test</h3>
          {budget.length ? (
            <table className="data">
              <thead>
                <tr><th>Budget</th><th>Mules caught</th><th>Recall</th><th>Precision</th></tr>
              </thead>
              <tbody>
                {budget.map((b: any) => (
                  <tr key={b.budget}>
                    <td>top {b.budget}</td>
                    <td>{b.true_positives} / {lt.n_positives}</td>
                    <td>{fmtPct(b.recall)}</td>
                    <td>{fmtPct(b.precision)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty msg="No budget simulation yet." />
          )}
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Open review queue (live)</h3>
          {cases.error ? <ErrorState msg={cases.error} /> :
            openCases.length ? (
            <table className="data">
              <thead><tr><th>Case</th><th>Tier</th><th>Risk</th></tr></thead>
              <tbody>
                {openCases.slice(0, 6).map((c) => (
                  <tr key={c.case_id}>
                    <td><a href={`#/cases/${c.case_id}`}>{c.case_id}</a></td>
                    <td><TierBadge tier={c.risk_tier} /></td>
                    <td>{fmtNum(c.calibrated_risk)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty msg="No open cases. Score accounts via the API or demo script to populate the queue." />
          )}
        </div>
        <div className="card">
          <h3>System status</h3>
          <table className="data">
            <tbody>
              <tr><td>Scoring service</td><td>{health.data ? <span className="badge ok">ready</span> : <span className="badge danger">down</span>}</td></tr>
              <tr><td>Model winner</td><td>{health.data?.winner ?? "–"}</td></tr>
              <tr><td>Local LLM (optional)</td><td>{health.data?.ollama_model ? <span className="badge ok">{health.data.ollama_model}</span> : <span className="badge warn">offline — deterministic narratives active</span>}</td></tr>
              <tr><td>Conformal abstention (locked test)</td><td>{fmtPct(lt?.conformal?.abstention_rate)}</td></tr>
              <tr><td>OOD rate (locked test)</td><td>{fmtPct(lt?.ood_rate, 2)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div className="footer-note">
        MuleGuard · Trinetra — Team Kryptonite · PS2 · Metrics: PR-AUC family under
        natural class prevalence (0.89% mules). Plain accuracy is deliberately not shown:
        a model that flags nothing would score ~99% accuracy while catching zero mules.
      </div>
    </>
  );
}
