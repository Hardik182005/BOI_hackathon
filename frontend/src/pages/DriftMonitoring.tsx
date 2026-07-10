import { api, fmtNum, fmtPct } from "../api";
import { Empty, ErrorState, Loading, usePoll } from "../components";

export default function DriftMonitoring() {
  const { data, error, loading } = usePoll(() => api.drift(), [], 20000);
  const metrics = usePoll(() => api.metrics(), []);
  if (loading) return <Loading what="drift status" />;
  if (error) return <ErrorState msg={error} />;

  const d = data?.latest;
  const registry = metrics.data?.registry ?? null;

  return (
    <>
      <h2 className="page-title">Drift &amp; Monitoring</h2>
      <p className="page-sub">
        PSI / distribution shift of inputs and scores against the frozen dev
        baseline. Bands (0.10 / 0.25) are stated conventions, not tuned results.
      </p>

      {!d ? (
        <Empty msg="No drift snapshot yet — the baseline is created at locked-test evaluation, and each scored batch adds a comparison." />
      ) : (
        <>
          <div className="grid cols-4">
            <div className="card">
              <h3>Overall status</h3>
              <div className="stat">
                <span className={`badge ${d.status === "STABLE" ? "ok" : d.status === "MODERATE" ? "warn" : "danger"}`}>
                  {d.status}
                </span>
              </div>
              <div className="stat-sub">{d.n_rows_scored} rows in latest batch</div>
            </div>
            <div className="card">
              <h3>Score distribution PSI</h3>
              <div className="stat">{fmtNum(d.score_psi, 4)}</div>
              <div className="stat-sub">calibrated risk vs dev baseline</div>
            </div>
            <div className="card">
              <h3>Features in alert band</h3>
              <div className="stat">{d.n_features_alert}</div>
              <div className="stat-sub">PSI &gt; {d.thresholds?.alert}</div>
            </div>
            <div className="card">
              <h3>Features in moderate band</h3>
              <div className="stat">{d.n_features_moderate}</div>
              <div className="stat-sub">PSI {d.thresholds?.moderate}–{d.thresholds?.alert}</div>
            </div>
          </div>
          <div className="card" style={{ marginTop: 14 }}>
            <h3>Highest-PSI features (latest batch)</h3>
            <table className="data">
              <thead><tr><th>Feature</th><th>PSI</th><th>Missingness shift</th></tr></thead>
              <tbody>
                {(d.worst_features ?? []).map((w: any) => (
                  <tr key={w.feature}>
                    <td>{w.feature}</td>
                    <td>{fmtNum(w.psi, 4)}</td>
                    <td>{w.missing_rate_shift >= 0 ? "+" : ""}{fmtPct(w.missing_rate_shift, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Champion / challenger governance</h3>
        <ol style={{ fontSize: 13, paddingLeft: 18, margin: "4px 0" }}>
          <li>A challenger model is trained on approved data.</li>
          <li>It is evaluated on the same immutable folds as the champion.</li>
          <li>Promotion is <b>never automatic</b> — an authorised approver records the decision in the model registry.</li>
          <li>Rollback: the previous bundle stays in <code>artifacts/model_registry</code> with its hash and can be reloaded at any time.</li>
        </ol>
        <div className="notice human">
          Analyst verdicts (confirmed mule / false positive) recorded on cases feed
          the retraining queue as fresh labels — closing the drift loop with human
          truth, not model self-labelling.
        </div>
      </div>
    </>
  );
}
