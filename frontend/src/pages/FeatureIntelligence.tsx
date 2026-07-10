import { useState } from "react";
import { api } from "../api";
import { Empty, ErrorState, Loading, usePoll } from "../components";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

// Served through the metrics endpoint? Feature artifacts come from static
// JSON served by the API's metrics summary; selection data is in lens/oof
// artifacts. We fetch the dedicated features file through the backend.
async function fetchFeatures() {
  const res = await fetch("/v1/metrics/summary");
  if (!res.ok) throw new Error("metrics unavailable");
  const m = await res.json();
  return m;
}

const VERIFIED: Record<string, string> = {
  F3886: "account/product type", F3888: "account opening date",
  F3889: "tenure/vintage bucket", F3890: "region/locality type",
  F3891: "occupation", F3892: "gender", F3893: "customer segment",
};

export default function FeatureIntelligence() {
  const { data, error, loading } = usePoll(() => api.metrics(), []);
  const [showAll, setShowAll] = useState(false);
  if (loading) return <Loading what="feature intelligence" />;
  if (error) return <ErrorState msg={error} />;

  const lens = data?.lens_stack_oof;
  const sel = data?.oof ? data?.selection ?? null : null;
  // selection frequency ships inside lens_stack payload? fall back to top reasons
  const freq: { feature: string; selection_frequency: number }[] =
    data?.selection_frequency ?? [];

  return (
    <>
      <h2 className="page-title">Feature Intelligence</h2>
      <p className="page-sub">
        Stability selection run inside training folds only (LightGBM gain + L1
        logistic over repeated subsamples). Anonymous features carry no invented
        business meaning.
      </p>

      <div className="notice">
        <b>Quarantined by the leakage firewall (never used by any model):</b>{" "}
        F3924 (target) · F3912 (measured target leak: |corr| 0.97, single-feature CV
        PR-AUC 0.94) · F2230 (snapshot month — all 9,001 negatives are Oct-2025,
        all 81 positives are Sep/Nov/Dec: the month reconstructs the label
        perfectly) · __UNNAMED__0 (row index; the file is physically ordered by label).
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Interpretable feature registry (verified from raw values)</h3>
          <table className="data">
            <thead><tr><th>Column</th><th>Verified meaning</th></tr></thead>
            <tbody>
              {Object.entries(VERIFIED).map(([f, name]) => (
                <tr key={f}><td>{f}</td><td>{name}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="notice">
            Every other Fxxxx column is an anonymised behavioural aggregate.
            Explanations state only how a value compares with the legitimate
            cohort — e.g. “F1702 is at the 99.2nd percentile and increases the
            model score” — never invented semantics like “proves laundering”.
          </div>
        </div>
        <div className="card">
          <h3>Selection stability</h3>
          {lens ? (
            <table className="data">
              <tbody>
                <tr><td>Winner model</td><td>{lens.winner}</td></tr>
                <tr><td>Compact set in production</td><td>top-60 stability-selected features</td></tr>
                <tr><td>Calibrator (OOF-selected)</td><td>{lens.calibration_selection?.winner}</td></tr>
                <tr><td>Hard negatives mined (dev OOF)</td><td>{lens.n_hard_negatives}</td></tr>
              </tbody>
            </table>
          ) : <Empty msg="Lens stack not built yet." />}
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Feature selection frequency (dev folds)</h3>
        {freq.length ? (
          <div style={{ width: "100%", height: 26 * Math.min(freq.length, showAll ? 60 : 20) + 50 }}>
            <ResponsiveContainer>
              <BarChart data={freq.slice(0, showAll ? 60 : 20)} layout="vertical"
                        margin={{ top: 4, right: 30, bottom: 4, left: 10 }}>
                <XAxis type="number" domain={[0, 1]} stroke="#5b6675" fontSize={11} />
                <YAxis type="category" dataKey="feature" width={90} stroke="#5b6675" fontSize={11} />
                <Tooltip contentStyle={{ background: "#fff", border: "1px solid #dde3ec" }}
                         formatter={(v: any) => [Number(v).toFixed(2), "selection frequency"]} />
                <ReferenceLine x={0.6} stroke="#94a3b8" strokeDasharray="4 4" />
                <ReferenceLine x={0.8} stroke="#94a3b8" strokeDasharray="2 4" />
                <Bar dataKey="selection_frequency" fill="#2563eb" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Empty msg="Selection-frequency artifact not exposed through the API yet — see artifacts/features/selection_frequency.csv and artifacts/plots/feature_stability.png." />
        )}
        {freq.length > 20 && (
          <button className="btn secondary" onClick={() => setShowAll(!showAll)}>
            {showAll ? "Show top 20" : "Show top 60"}
          </button>
        )}
      </div>
    </>
  );
}
