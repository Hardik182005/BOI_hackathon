import { useState } from "react";
import { api, fmtCi, fmtNum, fmtPct, lockedHeadline } from "../api";
import { Empty, ErrorState, HumanReviewNotice, Loading, RetiredArtifactNotice, usePoll } from "../components";

// Analyst-budget framing of the locked-test evaluation. Every figure on this
// page is a field of /v1/metrics/summary. The only arithmetic performed here is
// the subtraction printed in the column header of the budget table
// (budget − true positives), which is an identity between two API columns and
// is labelled as such - no metric is estimated in the browser.
//
// The one exception is the Impact Simulator, and it is an exception on purpose:
// it multiplies numbers the user typed by counts the artifact measured. It
// never turns an assumption into a metric - the recall it quotes is a row of
// the artifact, and every figure it prints names the assumption behind it.

export default function BusinessValue() {
  const { data, error, loading } = usePoll(() => api.metrics(), []);

  if (loading) return <Loading what="locked-test budget artifacts" />;
  if (error) return <ErrorState msg={error} />;

  const lt = data?.locked_test;
  if (!lt) {
    return (
      <>
        <h2 className="page-title">Business Value</h2>
        <Empty msg="The API responded, but no locked-test evaluation artifact is present, so there are no budget figures to show. Run the locked-test evaluation to populate this page." />
      </>
    );
  }

  const budgets: any[] = lt.recall_at_budget ?? [];
  const fpr: any[] = lt.recall_at_fpr ?? [];
  const tiers: any[] = lt.tier_distribution ?? [];
  const reviewTiers = tiers.filter((t) => t.tier !== "MONITOR");
  const head = lockedHeadline(lt);

  return (
    <>
      <h2 className="page-title">Business Value</h2>
      <p className="page-sub">
        What a review team actually gets for a fixed amount of analyst time.
        Split: {lt.split ?? "locked test"} · {lt.n?.toLocaleString()} accounts ·
        {" "}{lt.n_positives} true mules ({fmtPct(lt.prevalence, 2)} prevalence).
      </p>
      <HumanReviewNotice />
      <RetiredArtifactNotice head={head} />

      <div className="grid cols-4">
        <div className="card">
          <h3>Accounts in the evaluation</h3>
          <div className="stat">{lt.n?.toLocaleString() ?? "–"}</div>
          <div className="stat-sub">locked test, scored once</div>
        </div>
        <div className="card">
          <h3>True mules available to find</h3>
          <div className="stat">{lt.n_positives ?? "–"}</div>
          <div className="stat-sub">the ceiling for every budget below</div>
        </div>
        <div className="card">
          <h3>PR-AUC · locked test</h3>
          <div className="stat">{fmtNum(head.prAuc)}</div>
          <div className="stat-sub">
            no-skill baseline {fmtNum(lt.prevalence, 4)} · {fmtCi(head.prAucCi)}
          </div>
        </div>
        <div className="card">
          <h3>Scoring throughput</h3>
          <div className="stat">
            {lt.scoring_rows_per_second == null ? "–" : Math.round(lt.scoring_rows_per_second).toLocaleString()}
          </div>
          <div className="stat-sub">
            rows/second measured during the locked-test run
            {lt.scoring_runtime_seconds != null ? ` (${fmtNum(lt.scoring_runtime_seconds, 1)}s total)` : ""}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Mules caught at fixed analyst review budgets — locked test</h3>
        {budgets.length === 0 ? (
          <Empty msg="The locked-test artifact contains no recall_at_budget block, so no budget simulation can be shown." />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Review budget</th>
                  <th>Mules caught</th>
                  <th>Recall</th>
                  <th>Precision in the pile</th>
                  <th>Reviews that are not mules (budget − true positives)</th>
                  <th>Out of</th>
                </tr>
              </thead>
              <tbody>
                {budgets.map((b) => (
                  <tr key={b.budget}>
                    <td><b>top {b.budget}</b> accounts</td>
                    <td>{b.true_positives} of {lt.n_positives}</td>
                    <td>{fmtPct(b.recall)}</td>
                    <td>{fmtPct(b.precision)}</td>
                    <td>{b.budget - b.true_positives}</td>
                    <td>{lt.n?.toLocaleString()} scored accounts</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="notice">
          A review budget is the number of accounts a team can work through. The
          precision column is the API's own figure; the false-alarm column beside
          it is the arithmetic difference between the two API columns to its
          left, not a separately estimated number.
        </div>
      </div>

      <ImpactSimulator budgets={budgets} lt={lt} />

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>False-positive burden at fixed false-positive rates — locked test</h3>
          {fpr.length === 0 ? (
            <Empty msg="No recall_at_fpr block in the locked-test artifact." />
          ) : (
            <table className="data">
              <thead>
                <tr>
                  <th>FPR target</th><th>Achieved FPR</th>
                  <th>False alarms per 1,000 legitimate accounts</th><th>Recall</th>
                </tr>
              </thead>
              <tbody>
                {fpr.map((r) => (
                  <tr key={r.fpr_target}>
                    <td>{fmtPct(r.fpr_target, 2)}</td>
                    <td>{fmtPct(r.achieved_fpr, 2)}</td>
                    <td><b>{fmtNum(r.fp_per_1000_legit, 1)}</b></td>
                    <td>{fmtPct(r.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="notice">
            Every false positive is a real customer whose account is inspected
            without cause. The burden is reported here rather than buried, and it
            is the reason the system recommends review rather than any automatic
            restriction.
          </div>
        </div>

        <div className="card">
          <h3>Where the review effort lands — precision inside each tier</h3>
          {reviewTiers.length === 0 ? (
            <Empty msg="No tier distribution in the locked-test artifact." />
          ) : (
            <table className="data">
              <thead>
                <tr><th>Tier</th><th>Accounts routed</th><th>True mules</th><th>Precision in tier</th></tr>
              </thead>
              <tbody>
                {reviewTiers.map((t) => (
                  <tr key={t.tier}>
                    <td>{t.tier.replace(/_/g, " ")}</td>
                    <td>{t.n}</td>
                    <td>{t.n_true_mules}</td>
                    <td>{t.precision_in_tier == null ? "–" : fmtPct(t.precision_in_tier)}</td>
                  </tr>
                ))}
                {tiers.filter((t) => t.tier === "MONITOR").map((t) => (
                  <tr key={t.tier}>
                    <td>MONITOR — not currently flagged</td>
                    <td>{t.n}</td>
                    <td>{t.n_true_mules}</td>
                    <td>{t.precision_in_tier == null ? "–" : fmtPct(t.precision_in_tier)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="notice">
            The MONITOR row still contains {tiers.find((t) => t.tier === "MONITOR")?.n_true_mules ?? "–"}{" "}
            true mules on this split. Accounts below the review threshold are not
            certified as anything — they are simply not currently flagged.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Reading these numbers honestly</h3>
        <ul style={{ fontSize: 13, margin: 0, paddingLeft: 18 }}>
          <li>
            The budgets are evaluated on a single locked test split of{" "}
            {lt.n?.toLocaleString()} accounts containing {lt.n_positives} mules.
            A queue of the same proportion on a full bank book is a far larger
            number of reviews, and this page does not extrapolate to one.
          </li>
          <li>
            Recall at a budget is bounded by {lt.n_positives} — the number of
            true mules present in this split. Nothing here is projected onto a
            portfolio the model has not been evaluated on, and no monetary
            benefit is claimed, because the pipeline produces no such estimate.
          </li>
          <li>
            Plain accuracy is deliberately absent: at {fmtPct(lt.prevalence, 2)}{" "}
            prevalence, a model that flags nothing would be right about almost
            every account while finding no mules at all.
          </li>
        </ul>
      </div>

      <div className="footer-note">
        Source: /v1/metrics/summary → locked_test. No measured figure on this
        page is typed into the frontend, and the pipeline produces no monetary
        estimate, so none is claimed. The Impact Simulator is arithmetic over
        assumptions a user typed; it is labelled as such wherever it appears and
        is not a pipeline output.
      </div>
    </>
  );
}

// ------------------------------------------------------- Impact Simulator

// Four numbers a bank knows and this pipeline does not: how many alerts the
// desk can work, how long one review takes, what a confirmed mule costs it, and
// what an analyst hour costs. They are multiplied by measured counts and never
// the other way round - the recall and the mule count come from a row of the
// locked-test artifact, and a capacity that falls between two measured budgets
// is served by the largest measured budget below it rather than interpolated.

const inr = (x: number) => `₹${Math.round(x).toLocaleString("en-IN")}`;

type Assumption = { id: string; label: string; suffix: string; step?: string };

const ASSUMPTIONS: Assumption[] = [
  { id: "sim-alerts", label: "Alerts analysts can review per day", suffix: "accounts" },
  { id: "sim-minutes", label: "Analyst time per review", suffix: "minutes" },
  { id: "sim-exposure", label: "Assumed fraud exposure per confirmed mule", suffix: "₹", step: "1000" },
  { id: "sim-cost", label: "Analyst cost per hour", suffix: "₹/hour", step: "50" },
];

function ImpactSimulator({ budgets, lt }: { budgets: any[]; lt: any }) {
  const [alerts, setAlerts] = useState("50");
  const [minutes, setMinutes] = useState("20");
  const [exposure, setExposure] = useState("250000");
  const [cost, setCost] = useState("800");

  const values: Record<string, [string, (v: string) => void]> = {
    "sim-alerts": [alerts, setAlerts],
    "sim-minutes": [minutes, setMinutes],
    "sim-exposure": [exposure, setExposure],
    "sim-cost": [cost, setCost],
  };

  const controls = (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap" }}>
      {ASSUMPTIONS.map((a) => (
        <div key={a.id} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <label htmlFor={a.id} style={{ fontSize: 12, color: "var(--muted)" }}>
            {a.label} <span style={{ fontStyle: "italic" }}>({a.suffix})</span>
          </label>
          <input id={a.id} type="number" min={0} step={a.step ?? "1"}
                 style={{ width: 150 }} value={values[a.id][0]}
                 onChange={(e) => values[a.id][1](e.target.value)} />
        </div>
      ))}
    </div>
  );

  const header = (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
      <h3 style={{ margin: 0 }}>Impact Simulator — what-if over assumptions you enter</h3>
      <span className="badge info">user-entered assumptions, not measured facts</span>
    </div>
  );

  const nums = [alerts, minutes, exposure, cost].map((v) => Number.parseFloat(v));
  const [nAlerts, nMinutes, nExposure, nCost] = nums;
  if (nums.some((v) => !Number.isFinite(v) || v < 0)) {
    return (
      <div className="card" style={{ marginTop: 14 }}>
        {header}
        <div style={{ marginTop: 12 }}>{controls}</div>
        <Empty msg="Enter a number of zero or more in every field. Nothing is estimated from a blank or negative assumption." />
      </div>
    );
  }

  // The measured row this capacity is served by: the largest budget the
  // artifact actually evaluated that a desk of this size can work through.
  const ordered = [...budgets].sort((a, b) => a.budget - b.budget);
  const row = ordered.filter((b) => b.budget <= nAlerts).pop();
  const largest = ordered[ordered.length - 1];

  const hours = row ? (row.budget * nMinutes) / 60 : 0;
  const caught = row?.true_positives ?? 0;
  const missed = (lt.n_positives ?? 0) - caught;

  return (
    <div className="card" style={{ marginTop: 14 }}>
      {header}
      <div style={{ marginTop: 12 }}>{controls}</div>

      {!row ? (
        <Empty msg={`The smallest budget the locked-test artifact evaluated is top ${ordered[0]?.budget ?? "–"} accounts, so a desk reviewing ${nAlerts} per day is below every measured point. The recall of a smaller queue is not extrapolated here.`} />
      ) : (
        <>
          <div className="grid cols-4" style={{ marginTop: 14 }}>
            <div className="card">
              <h3>Mules found at this capacity</h3>
              <div className="stat">{caught} of {lt.n_positives}</div>
              <div className="stat-sub">
                measured — recall {fmtPct(row.recall)} at top {row.budget} on the
                locked test
              </div>
            </div>
            <div className="card">
              <h3>Review hours per scored batch</h3>
              <div className="stat">{fmtNum(hours, 1)}</div>
              <div className="stat-sub">
                your assumption — {row.budget} reviews × {nMinutes} minutes
              </div>
            </div>
            <div className="card">
              <h3>Analyst cost of that review load</h3>
              <div className="stat">{inr(hours * nCost)}</div>
              <div className="stat-sub">
                your assumption — {fmtNum(hours, 1)} hours × {inr(nCost)} per hour
              </div>
            </div>
            <div className="card">
              <h3>Hypothetical exposure surfaced</h3>
              <div className="stat">{inr(caught * nExposure)}</div>
              <div className="stat-sub">
                your assumption — {caught} mules found × {inr(nExposure)} each
              </div>
            </div>
          </div>

          <table className="kv" style={{ marginTop: 14 }}>
            <tbody>
              <tr>
                <td>Measured row this answer stands on</td>
                <td>
                  top {row.budget} of {lt.n?.toLocaleString()} accounts —
                  recall {fmtPct(row.recall)}, precision {fmtPct(row.precision)},
                  {" "}{row.budget - caught} reviews that are not mules.
                  {nAlerts > largest.budget
                    ? ` Your ${nAlerts} per day is above the largest budget the artifact evaluated (${largest.budget}); the answer is held at that budget rather than extrapolated.`
                    : nAlerts > row.budget
                    ? ` Your ${nAlerts} per day is served by the largest measured budget at or below it; recall is never interpolated between measured budgets.`
                    : ""}
                </td>
              </tr>
              <tr>
                <td>Exposure attached to the mules this capacity does not reach</td>
                <td>
                  {inr(missed * nExposure)} — {missed} of {lt.n_positives} mules
                  in this split are below the line at top {row.budget}. The same
                  assumption that produces the figure above produces this one,
                  and a simulator that showed only the first would be selling
                  rather than reporting.
                </td>
              </tr>
              <tr>
                <td>What this is not</td>
                <td>
                  Not a saving, not a forecast, and not a pipeline output. Four
                  of the six numbers above are yours; the pipeline contributed
                  only the mule count and the recall at a measured budget on one
                  split of {lt.n?.toLocaleString()} accounts.
                </td>
              </tr>
            </tbody>
          </table>

          <div className="notice human">
            Exposure per mule and analyst cost are assumptions this system cannot
            measure and does not hold: the dataset carries no monetary loss
            field. Change either input and every rupee figure here changes with
            it, which is the point of the panel — it prices your assumption, not
            the model.
          </div>
        </>
      )}
    </div>
  );
}
