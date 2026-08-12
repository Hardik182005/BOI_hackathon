import { useState } from "react";
import { api, fmtNum, fmtPct, Json } from "../api";
import { Empty, ErrorState, HumanReviewNotice, Loading, usePoll } from "../components";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

// Analyst Capacity Optimizer.
//
// The judge types the one number a head of operations actually knows - how many
// accounts the desk can review - or the one a compliance officer knows - how
// many false alarms per 1,000 accounts is tolerable - and the page reports what
// the measured out-of-fold curve says about it.
//
// Not one figure on this page is computed in the browser. /v1/capacity/curve
// serves a precomputed artifact and /v1/capacity/plan is a lookup inside it;
// the only arithmetic here is turning a fraction into a percentage for display.
// The recommended threshold is rendered as a proposal, with the frozen policy
// version beside it, because approving it is a human's job.

const tooltipStyle = { background: "#ffffff", border: "1px solid #dde3ec", color: "#111" };
const PRESETS = [25, 50, 100];

type Mode = "capacity" | "fp";

export default function CapacityOptimizer() {
  const { data: curve, error, loading } = usePoll(() => api.capacityCurve(), []);
  const [mode, setMode] = useState<Mode>("capacity");
  const [capacityInput, setCapacityInput] = useState("50");
  const [fpInput, setFpInput] = useState("5");
  const [basis, setBasis] = useState("legitimate");
  const [plan, setPlan] = useState<Json | null>(null);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (loading) return <Loading what="the measured capacity curve" />;
  if (error) return <ErrorState msg={error} />;
  if (!curve) return <Empty msg="The API returned no capacity curve." />;

  const ev = curve.evaluation ?? {};
  const prov = curve.provenance ?? {};
  const points: any[] = curve.points ?? [];
  const headline: any[] = curve.headline ?? [];
  const position: any[] = curve.frozen_policy_position ?? [];

  const ask = async (body: Json) => {
    setBusy(true);
    setPlanErr(null);
    try {
      setPlan(await api.capacityPlan(body));
    } catch (e: any) {
      setPlan(null);
      setPlanErr(e.message ?? "the capacity question could not be answered");
    } finally {
      setBusy(false);
    }
  };

  const submit = () => {
    if (mode === "capacity") {
      const k = Number.parseInt(capacityInput, 10);
      if (!Number.isFinite(k) || k < 1) {
        setPlanErr("enter a review capacity of at least 1 account");
        return;
      }
      void ask({ review_capacity: k });
    } else {
      const f = Number.parseFloat(fpInput);
      if (!Number.isFinite(f) || f < 0) {
        setPlanErr("enter a false-alarm tolerance of 0 or more");
        return;
      }
      void ask({ max_fp_per_1000: f, fp_basis: basis });
    }
  };

  // Display-only: the chart shows the measured points inside the range a
  // review desk operates in. No point is interpolated or invented - these are
  // the same rows the table below prints.
  const chart = points
    .filter((p) => p.budget <= 400)
    .map((p) => ({
      budget: p.budget,
      recall: p.recall,
      lo: p.recall_ci_low_resampled_accounts,
      hi: p.recall_ci_high_resampled_accounts,
      precision: p.precision,
    }));

  const exp = plan?.expected;
  const rec = plan?.recommended_threshold;

  return (
    <>
      <h2 className="page-title">Analyst Capacity Optimizer</h2>
      <p className="page-sub">
        What a fixed amount of analyst time buys, measured — not assumed.
        Source: {ev.split} · {ev.n_accounts?.toLocaleString()} accounts ·{" "}
        {ev.n_positives} confirmed mules ({fmtPct(ev.prevalence, 2)} prevalence) ·
        champion {prov.champion_model} · no retraining.
      </p>
      <HumanReviewNotice />

      <div className="card">
        <h3>Investigation capacity — ask one question</h3>
        <div className="tabs">
          <button className={mode === "capacity" ? "active" : ""}
                  onClick={() => { setMode("capacity"); setPlan(null); setPlanErr(null); }}>
            Accounts analysts can review per day
          </button>
          <button className={mode === "fp" ? "active" : ""}
                  onClick={() => { setMode("fp"); setPlan(null); setPlanErr(null); }}>
            Maximum acceptable false positives per 1,000
          </button>
        </div>

        {mode === "capacity" ? (
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <label htmlFor="cap-input" style={{ fontSize: 13 }}>
              Accounts reviewable per day
            </label>
            <input id="cap-input" type="number" min={1} style={{ width: 120 }}
                   value={capacityInput}
                   onChange={(e) => setCapacityInput(e.target.value)} />
            {PRESETS.map((k) => (
              <button key={k} className="btn secondary"
                      onClick={() => { setCapacityInput(String(k)); void ask({ review_capacity: k }); }}>
                {k}
              </button>
            ))}
            <button className="btn" disabled={busy} onClick={submit}>
              {busy ? "Reading the curve…" : "Show what that buys"}
            </button>
          </div>
        ) : (
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <label htmlFor="fp-input" style={{ fontSize: 13 }}>
              Maximum false positives per 1,000
            </label>
            <input id="fp-input" type="number" min={0} step="0.1" style={{ width: 120 }}
                   value={fpInput} onChange={(e) => setFpInput(e.target.value)} />
            <select aria-label="false positive denominator" value={basis}
                    onChange={(e) => setBasis(e.target.value)}>
              <option value="legitimate">per 1,000 legitimate accounts</option>
              <option value="screened">per 1,000 accounts screened</option>
            </select>
            <button className="btn" disabled={busy} onClick={submit}>
              {busy ? "Reading the curve…" : "Show the largest budget inside it"}
            </button>
          </div>
        )}
        <div className="notice">
          One question at a time, so that it is always clear which constraint is
          binding. Nothing here is a cost function: the panel never asks the bank
          to price a missed mule against a false alarm, it reports what the
          measured ranking does at the capacity the bank already has.
        </div>
        {planErr && <ErrorState msg={planErr} />}
      </div>

      {plan && exp && (
        <>
          <div className="grid cols-4" style={{ marginTop: 14 }}>
            <div className="card">
              <h3>Expected recall at {plan.answered_at_budget} reviews</h3>
              <div className="stat">{fmtPct(exp.recall)}</div>
              <div className="stat-sub">
                {exp.mules_caught} of {exp.mules_available} mules in this split ·
                95% interval {fmtPct(exp.recall_ci_low_resampled_accounts)}–
                {fmtPct(exp.recall_ci_high_resampled_accounts)}
              </div>
            </div>
            <div className="card">
              <h3>Precision in the review pile</h3>
              <div className="stat">{fmtPct(exp.precision)}</div>
              <div className="stat-sub">
                {exp.false_alarms} of {plan.answered_at_budget} reviews are not
                mules · 95% interval {fmtPct(exp.precision_ci_low)}–{fmtPct(exp.precision_ci_high)}
              </div>
            </div>
            <div className="card">
              <h3>False alarms per 1,000</h3>
              <div className="stat">{fmtNum(exp.fp_per_1000_legitimate, 2)}</div>
              <div className="stat-sub">
                per 1,000 legitimate accounts ·{" "}
                {fmtNum(exp.fp_per_1000_screened, 2)} per 1,000 screened
              </div>
            </div>
            <div className="card">
              <h3>Mules this budget misses</h3>
              <div className="stat">{exp.mules_missed}</div>
              <div className="stat-sub">
                out of {exp.mules_available}. Accounts below the line are not
                cleared — they stay under monitoring.
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <h3>Recommended operational threshold</h3>
            <div style={{ display: "flex", gap: 14, alignItems: "baseline", flexWrap: "wrap" }}>
              <div className="stat" style={{ fontSize: 22 }}>
                {rec?.calibrated_risk == null ? "–" : rec.calibrated_risk.toFixed(6)}
              </div>
              <span className="badge warn">{rec?.status?.replace(/_/g, " ")}</span>
              <span style={{ fontSize: 12.5, color: "var(--muted)" }}>
                applied: {String(rec?.applied)}
              </span>
            </div>
            <table className="kv" style={{ marginTop: 10 }}>
              <tbody>
                <tr>
                  <td>Scale</td>
                  <td>{rec?.scale}</td>
                </tr>
                <tr>
                  <td>Alerts it produces on the evaluation split</td>
                  <td>
                    {rec?.alerts_produced_on_evaluation_split} — against a
                    requested budget of {plan.answered_at_budget}
                  </td>
                </tr>
                <tr>
                  <td>Position under the frozen policy</td>
                  <td>
                    {rec?.band_under_frozen_policy} (policy version{" "}
                    {rec?.frozen_policy_version})
                  </td>
                </tr>
                <tr>
                  <td>Underlying model score at that rank</td>
                  <td className="mono">{fmtNum(rec?.model_score, 6)}</td>
                </tr>
              </tbody>
            </table>
            <div className="notice human">{rec?.note}</div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <h3>How firm is this number?</h3>
            <table className="kv">
              <tbody>
                <tr>
                  <td>Recall, ranking held to {exp.mules_available} mules</td>
                  <td>
                    {fmtPct(exp.recall_ci_low)}–{fmtPct(exp.recall_ci_high)}
                    {exp.stratified_interval_degenerate && (
                      <> — collapsed onto the point estimate, because this budget sits
                      inside the leading run of {ev.leading_true_positive_run} accounts
                      that are all mules, so there is nothing for this scheme to vary</>
                    )}
                  </td>
                </tr>
                <tr>
                  <td>Recall, whole book resampled</td>
                  <td>
                    {fmtPct(exp.recall_ci_low_resampled_accounts)}–
                    {fmtPct(exp.recall_ci_high_resampled_accounts)} — the wider,
                    more pessimistic reading, and the one to quote
                  </td>
                </tr>
                <tr>
                  <td>Recall in each individual CV repeat</td>
                  <td>
                    {(exp.recall_per_repeat ?? []).map((r: number) => fmtPct(r)).join(" · ")}
                    {" "}(range {fmtPct(exp.recall_repeat_min)}–{fmtPct(exp.recall_repeat_max)})
                  </td>
                </tr>
                <tr>
                  <td>One mule is worth</td>
                  <td>
                    {fmtNum(ev.recall_resolution_pct_points, 2)} percentage points
                    of recall
                  </td>
                </tr>
              </tbody>
            </table>
            <div className="notice">{plan.uncertainty_note}</div>
          </div>
        </>
      )}

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Measured capacity curve — recall against review budget</h3>
          {chart.length ? (
            <div className="chart-box">
              <ResponsiveContainer>
                <LineChart data={chart} margin={{ top: 6, right: 12, bottom: 12, left: 0 }}>
                  <XAxis dataKey="budget" stroke="#5b6675" fontSize={11} type="number"
                         domain={["dataMin", "dataMax"]}
                         label={{ value: "accounts reviewed", position: "insideBottom", offset: -6, fontSize: 11 }} />
                  <YAxis stroke="#5b6675" fontSize={11} domain={[0, 1]} />
                  <Tooltip contentStyle={tooltipStyle}
                           formatter={(v: any, n: any) => [Number(v).toFixed(3), n]} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="recall" name="recall" dot={false} stroke="#2563eb" strokeWidth={2} />
                  <Line dataKey="lo" name="95% low" dot={false} stroke="#94a3b8"
                        strokeDasharray="4 4" strokeWidth={1} />
                  <Line dataKey="hi" name="95% high" dot={false} stroke="#94a3b8"
                        strokeDasharray="4 4" strokeWidth={1} />
                  <Line dataKey="precision" name="precision" dot={false} stroke="#d97706"
                        strokeWidth={1.5} />
                  {plan && (
                    <ReferenceLine x={plan.answered_at_budget} stroke="#111"
                                   strokeDasharray="3 3"
                                   label={{ value: `budget ${plan.answered_at_budget}`, fontSize: 10, fill: "#111" }} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <Empty msg="The curve artifact contains no points." />}
          <div className="notice">
            Recall climbs steeply and then flattens: the first few dozen reviews
            are worth far more than the next few hundred. Precision falls the
            whole way, which is the cost of that extra recall paid in customers
            inspected without cause.
          </div>
        </div>

        <div className="card">
          <h3>Where the frozen policy already sits on this curve</h3>
          {position.length ? (
            <table className="data">
              <thead>
                <tr>
                  <th>Review tier</th><th>Threshold</th><th>Accounts</th>
                  <th>Mules found</th><th>Recall</th><th>Precision</th>
                </tr>
              </thead>
              <tbody>
                {position.map((p) => (
                  <tr key={p.tier}>
                    <td>{p.tier.replace(/_/g, " ")}</td>
                    <td className="mono">{fmtNum(p.calibrated_risk_threshold, 5)}</td>
                    <td>{p.accounts_at_or_above}</td>
                    <td>{p.mules_caught}</td>
                    <td>{fmtPct(p.recall)}</td>
                    <td>{fmtPct(p.precision)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty msg="No frozen-policy position in the artifact." />}
          <div className="notice">
            Policy version {curve.frozen_policy?.policy_version} is itself a
            capacity decision, taken before this panel existed. These rows are
            read from the frozen thresholds; this page compares against them and
            never rewrites them.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Review budgets at a glance</h3>
        {headline.length ? (
          <div style={{ overflowX: "auto" }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Review budget</th><th>Expected recall</th>
                  <th>95% interval (book resampled)</th><th>Per CV repeat</th>
                  <th>Precision</th><th>Mules caught</th><th>False alarms</th>
                  <th>Advisory threshold</th>
                </tr>
              </thead>
              <tbody>
                {headline.map((p) => (
                  <tr key={p.budget}>
                    <td><b>{p.budget}</b> accounts</td>
                    <td><b>{fmtPct(p.recall)}</b></td>
                    <td>
                      {fmtPct(p.recall_ci_low_resampled_accounts)}–
                      {fmtPct(p.recall_ci_high_resampled_accounts)}
                    </td>
                    <td>{fmtPct(p.recall_repeat_min)}–{fmtPct(p.recall_repeat_max)}</td>
                    <td>{fmtPct(p.precision)}</td>
                    <td>{p.true_positives} of {ev.n_positives}</td>
                    <td>{p.false_positives}</td>
                    <td className="mono">{fmtNum(p.threshold_calibrated_risk, 5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Empty msg="No headline budgets in the artifact." />}
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>What this number does not mean</h3>
        <ul style={{ fontSize: 13, margin: 0, paddingLeft: 18 }}>
          <li>
            It is not a promise. It is what the champion's stored out-of-fold
            predictions did on {ev.n_accounts?.toLocaleString()} development
            accounts holding {ev.n_positives} mules. A different month, a
            different fraud pattern, or a different book will move it.
          </li>
          <li>
            It is not a monetary saving. The pipeline produces no loss estimate,
            so this page shows none.
          </li>
          <li>
            The accounts below the threshold are not cleared. They are simply
            not currently flagged, and monitoring continues on them.
          </li>
          <li>
            The recommended threshold is advisory. Adopting it is a documented
            human decision, and nothing punitive follows automatically from it.
          </li>
          {(prov.caveats ?? []).map((c: string, i: number) => <li key={i}>{c}</li>)}
        </ul>
      </div>

      <div className="footer-note">
        Source: /v1/capacity/curve and /v1/capacity/plan, both served from{" "}
        {prov.predictions_source} via {prov.generator}. Champion read from{" "}
        {prov.champion_source}; calibrator {prov.calibrator} from{" "}
        {prov.calibrator_source}. Retraining performed:{" "}
        {String(prov.retraining_performed)}. Bootstrap{" "}
        {prov.bootstrap?.n_boot?.toLocaleString()} replicates, seed {prov.seed}.
        No figure on this page is typed into the frontend.
      </div>
    </>
  );
}
