import { api, fmtCi, fmtNum, fmtPct, lockedHeadline } from "../api";
import { Empty, ErrorState, Loading, RetiredArtifactNotice, usePoll } from "../components";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine, Legend,
} from "recharts";

const tooltipStyle = { background: "#ffffff", border: "1px solid #dde3ec", color: "#111" };

export default function ModelPerformance() {
  const { data, error, loading } = usePoll(() => api.metrics(), []);
  if (loading) return <Loading what="model performance artifacts" />;
  if (error) return <ErrorState msg={error} />;

  const lt = data?.locked_test;
  const head = lockedHeadline(lt);
  const oof = data?.oof?.models ?? {};
  const ablation = data?.leakage_ablation;
  const ensemble = data?.ensemble_decision;

  const prCurve = lt
    ? lt.pr_curve.recall.map((r: number, i: number) => ({
        recall: r, precision: lt.pr_curve.precision[i],
      }))
    : [];
  const calBins = (lt?.calibration?.bins ?? []).map((b: any) => ({
    predicted: b.mean_predicted, observed: b.observed_rate, count: b.count,
  }));
  const comparison = Object.entries(oof)
    .map(([name, m]: any) => ({
      name, pr: m.pr_auc_mean, std: m.pr_auc_std,
      rejected: name.startsWith("REJECTED"),
    }))
    .sort((a, b) => a.pr - b.pr);

  return (
    <>
      <h2 className="page-title">Model Performance</h2>
      <p className="page-sub">
        Primary metric: PR-AUC (Average Precision) at natural prevalence.
        Dev = repeated stratified 5-fold out-of-fold; locked test touched once.
      </p>
      <RetiredArtifactNotice head={head} />

      <div className="grid cols-4">
        <div className="card">
          <h3>PR-AUC · locked test</h3>
          <div className="stat">{fmtNum(head.prAuc)}</div>
          <div className="stat-sub">
            {head.fromDeployedModel
              ? <>{fmtCi(head.prAucCi)} · bootstrap n=2000</>
              : <>deployed scorer on the sealed split · {fmtCi(head.prAucCi)}</>}
          </div>
        </div>
        <div className="card">
          <h3>ROC-AUC (secondary)</h3>
          <div className="stat">{fmtNum(head.rocAuc)}</div>
          <div className="stat-sub">reported for context only — misleading alone at 0.9% prevalence</div>
        </div>
        <div className="card">
          <h3>Brier / ECE</h3>
          <div className="stat">{lt?.brier != null ? lt.brier.toFixed(5) : "–"}</div>
          <div className="stat-sub">ECE {fmtNum(lt?.calibration?.ece, 4)} · locked test calibrated probabilities</div>
        </div>
        <div className="card">
          <h3>Ensemble decision</h3>
          <div className="stat" style={{ fontSize: 16 }}>
            {ensemble ? (ensemble.accepted
              ? <span className="badge info">stacker accepted</span>
              : <span className="badge ok">best single model kept</span>) : "–"}
          </div>
          <div className="stat-sub">{ensemble?.rule}</div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Precision-Recall curve — locked test</h3>
          {prCurve.length ? (
            <div className="chart-box">
              <ResponsiveContainer>
                <LineChart data={prCurve} margin={{ top: 6, right: 10, bottom: 4, left: 0 }}>
                  <XAxis dataKey="recall" stroke="#5b6675" fontSize={11}
                         label={{ value: "recall", position: "insideBottom", offset: -2, fontSize: 11 }} />
                  <YAxis stroke="#5b6675" fontSize={11} domain={[0, 1]} />
                  <Tooltip contentStyle={tooltipStyle}
                           formatter={(v: any, n: any) => [Number(v).toFixed(3), n]} />
                  <ReferenceLine y={lt.prevalence} stroke="#94a3b8" strokeDasharray="4 4"
                                 label={{ value: `no-skill ${lt.prevalence.toFixed(4)}`, fontSize: 10, fill: "#5b6675" }} />
                  <Line dataKey="precision" dot={false} stroke="#2563eb" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <Empty msg="Locked-test evaluation not run yet." />}
        </div>

        <div className="card">
          <h3>Calibration — locked test</h3>
          {calBins.length ? (
            <div className="chart-box">
              <ResponsiveContainer>
                <LineChart data={calBins} margin={{ top: 6, right: 10, bottom: 4, left: 0 }}>
                  <XAxis dataKey="predicted" stroke="#5b6675" fontSize={11} domain={[0, 1]}
                         type="number" label={{ value: "mean predicted risk", position: "insideBottom", offset: -2, fontSize: 11 }} />
                  <YAxis stroke="#5b6675" fontSize={11} domain={[0, 1]} />
                  <Tooltip contentStyle={tooltipStyle}
                           formatter={(v: any, n: any) => [Number(v).toFixed(3), n]} />
                  <Line dataKey="observed" stroke="#2563eb" strokeWidth={2} />
                  <Line data={[{ predicted: 0, observed: 0 }, { predicted: 1, observed: 1 }]}
                        dataKey="observed" stroke="#94a3b8" strokeDasharray="4 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <Empty msg="No calibration bins yet." />}
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Model tournament — dev OOF PR-AUC (mean ± std over repeats)</h3>
        {comparison.length ? (
          <div style={{ width: "100%", height: 30 * comparison.length + 60 }}>
            <ResponsiveContainer>
              <BarChart data={comparison} layout="vertical"
                        margin={{ top: 4, right: 60, bottom: 4, left: 10 }}>
                <XAxis type="number" domain={[0, 1.05]} stroke="#5b6675" fontSize={11} />
                <YAxis type="category" dataKey="name" width={230} stroke="#5b6675" fontSize={11} />
                <Tooltip contentStyle={tooltipStyle}
                         formatter={(v: any) => [Number(v).toFixed(4), "PR-AUC"]} />
                <Bar dataKey="pr" radius={[0, 4, 4, 0]}>
                  {comparison.map((d) => (
                    <Cell key={d.name} fill={d.rejected ? "#dc2626" : "#2563eb"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : <Empty msg="No tournament results yet." />}
        {ablation && (
          <div className="notice" style={{ borderColor: "#dc2626" }}>
            <b>Leakage panel:</b> the red bar (
            <span className="badge rejected">REJECTED LEAKAGE</span>) is the run that
            re-admits quarantined feature F3912 — PR-AUC {fmtNum(ablation.with_f3912?.pr_auc_mean)} vs
            clean {fmtNum(ablation.without_f3912?.pr_auc_mean)}. It is evidence of why the
            quarantine exists, never a model result. {ablation.note}
          </div>
        )}
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Recall at fixed false-positive rates — locked test</h3>
          {lt?.recall_at_fpr?.length ? (
            <table className="data">
              <thead><tr><th>FPR target</th><th>Recall</th><th>FP / 1,000 legit</th><th>Threshold</th></tr></thead>
              <tbody>
                {lt.recall_at_fpr.map((r: any) => (
                  <tr key={r.fpr_target}>
                    <td>{fmtPct(r.fpr_target, 1)}</td>
                    <td>{fmtPct(r.recall)}</td>
                    <td>{r.fp_per_1000_legit?.toFixed(1)}</td>
                    <td>{fmtNum(r.threshold)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty msg="No FPR table yet." />}
        </div>
        <div className="card">
          <h3>Precision inside review tiers — locked test</h3>
          {lt?.tier_distribution ? (
            <table className="data">
              <thead><tr><th>Tier</th><th>Accounts</th><th>True mules</th><th>Precision in tier</th></tr></thead>
              <tbody>
                {lt.tier_distribution.map((t: any) => (
                  <tr key={t.tier}>
                    <td>{t.tier}</td><td>{t.n}</td><td>{t.n_true_mules}</td>
                    <td>{t.precision_in_tier == null ? "–" : fmtPct(t.precision_in_tier)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty msg="Tier distribution appears after locked-test evaluation." />}
        </div>
      </div>
      <div className="footer-note">
        No plain-accuracy trophy number is displayed anywhere: at 0.89% prevalence a
        do-nothing model reaches ~99% accuracy while catching zero mules.
      </div>
    </>
  );
}
