import { api, fmtCi, fmtNum, fmtPct, lockedHeadline } from "../api";
import { Empty, ErrorState, HumanReviewNotice, Loading, RetiredArtifactNotice, usePoll } from "../components";

// Analyst-budget framing of the locked-test evaluation. Every figure on this
// page is a field of /v1/metrics/summary. The only arithmetic performed here is
// the subtraction printed in the column header of the budget table
// (budget − true positives), which is an identity between two API columns and
// is labelled as such - no metric is estimated in the browser.

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
        Source: /v1/metrics/summary → locked_test. No figure on this page is
        typed into the frontend, and no projection or monetary estimate is shown,
        because the pipeline produces none.
      </div>
    </>
  );
}
