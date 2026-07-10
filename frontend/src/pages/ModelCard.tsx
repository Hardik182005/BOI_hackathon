import { api, fmtNum } from "../api";
import { ErrorState, Loading, usePoll } from "../components";

export default function ModelCard() {
  const model = usePoll(() => api.model(), []);
  const metrics = usePoll(() => api.metrics(), []);
  if (model.loading) return <Loading what="model card" />;

  const m = model.data;
  const lt = metrics.data?.locked_test;

  return (
    <>
      <h2 className="page-title">Model Card — MuleGuard · Trinetra</h2>
      <p className="page-sub">
        Purpose, data, exclusions, limits and safety constraints of the deployed
        scoring bundle. Values load from the signed model manifest.
      </p>
      {model.error && <ErrorState msg={model.error} />}

      <div className="grid cols-2">
        <div className="card">
          <h3>Purpose</h3>
          <p style={{ fontSize: 13.5 }}>
            Rank bank accounts by behavioural similarity to labelled mule
            accounts so human analysts can review the highest-risk cases first.
            The model output is a <b>calibrated behavioural risk score</b> — it is
            not a fraud verdict, not proof of intent, and never triggers
            automatic punitive action.
          </p>
          <h3>Training data</h3>
          <p style={{ fontSize: 13.5 }}>
            Provided portal dataset: 9,082 accounts × 3,923 features, 81
            labelled mules (0.89% prevalence). Development split of 7,264
            accounts (64 positives); locked test of 1,818 accounts (17
            positives) touched exactly once. Raw file SHA-256 fingerprint:
            <code style={{ fontSize: 11 }}> {m?.data_fingerprint_sha256?.slice(0, 24)}…</code>
          </p>
        </div>
        <div className="card">
          <h3>Current version</h3>
          <table className="data">
            <tbody>
              <tr><td>Bundle version</td><td>{m?.winner ?? "–"} v{m ? "1.0.0" : "–"}</td></tr>
              <tr><td>Bundle SHA-256</td><td style={{ fontSize: 11 }}>{m?.bundle_sha256?.slice(0, 32)}…</td></tr>
              <tr><td>Features used</td><td>{m?.n_features} (stability-selected, compact)</td></tr>
              <tr><td>Calibrator</td><td>{m?.calibrator}</td></tr>
              <tr><td>Git commit</td><td style={{ fontSize: 11 }}>{m?.git?.commit_sha?.slice(0, 12)}</td></tr>
              <tr><td>Dev OOF PR-AUC</td><td>{fmtNum(m?.oof_pr_auc)}</td></tr>
              <tr><td>Locked-test PR-AUC</td><td>{fmtNum(lt?.pr_auc?.point)} (CI {fmtNum(lt?.pr_auc?.ci_low)}–{fmtNum(lt?.pr_auc?.ci_high)})</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Exclusions (leakage firewall)</h3>
          <ul style={{ fontSize: 13, paddingLeft: 18 }}>
            <li><b>F3924</b> — the target itself.</li>
            <li><b>F3912</b> — measured target leak (|corr| 0.97, single-feature CV PR-AUC 0.94).</li>
            <li><b>F2230</b> — snapshot month; separates classes perfectly by dataset construction (all negatives Oct-2025, all positives Sep/Nov/Dec).</li>
            <li><b>__UNNAMED__0</b> — row index; the file is ordered by label.</li>
            <li>In-fold: constant columns and exact duplicates removed on training statistics only.</li>
          </ul>
        </div>
        <div className="card">
          <h3>Ethical &amp; safety constraints</h3>
          <ul style={{ fontSize: 13, paddingLeft: 18 }}>
            <li>No automatic freezing — every action requires an authorised analyst with a recorded reason.</li>
            <li>"Low risk" means <i>not currently flagged; monitoring continues</i> — never "certified safe".</li>
            <li>Uncertain and out-of-distribution inputs route to human review instead of receiving a confident label.</li>
            <li>The local LLM can only narrate verified facts; its output is machine-validated and any deviation is discarded for a deterministic template.</li>
            <li>Anonymous features are never given invented business meanings.</li>
            <li>Append-only audit trail for every score, decision and report.</li>
          </ul>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Known limitations</h3>
        <ul style={{ fontSize: 13, paddingLeft: 18 }}>
          <li>The dataset is a flat per-account snapshot: no transaction sequences and no counterparty graph, so no network/ring detection is claimed (roadmap item once counterparty data exists).</li>
          <li>Behaviour is not intent: the model cannot distinguish a wilful mule from a hacked or coerced victim — that distinction belongs to analysts with enrichment data.</li>
          <li>Dormant (sleeper) mules with no behavioural footprint are not claimed detectable before activation.</li>
          <li>With 81 total positives, confidence intervals are wide; they are reported, not hidden.</li>
          <li>The label collection is confounded with snapshot month, so an out-of-time stress test would be meaningless on this file (time = label); documented instead of faked.</li>
        </ul>
      </div>
    </>
  );
}
