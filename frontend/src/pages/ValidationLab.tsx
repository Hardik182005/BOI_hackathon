import { useState } from "react";
import { api, fmtNum, fmtPct } from "../api";
import { Empty, ErrorState, Loading, usePoll } from "../components";

// The judge-facing hidden-validation workflow. Two deliberately separate
// actions: RUN seals predictions and reports no metric at all, REVEAL asks for
// labels and only then reports numbers. The page keeps that separation
// visible - the metric panel says, in words, that no metric exists yet.

const VERDICTS: Record<string, string> = { PASS: "PASS", WARN: "WARN", FAIL: "FAIL" };

function Verdict({ v }: { v: string }) {
  return <span className={`verdict ${VERDICTS[v] ?? "WARN"}`}>{v}</span>;
}

export default function ValidationLab() {
  const [file, setFile] = useState<File | null>(null);
  const [labelFile, setLabelFile] = useState<File | null>(null);
  const [labelColumn, setLabelColumn] = useState("");
  const [run, setRun] = useState<any>(null);
  const [runErr, setRunErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [reveal, setReveal] = useState<any>(null);
  const [revealErr, setRevealErr] = useState<string | null>(null);
  const [sealRefresh, setSealRefresh] = useState(0);

  const seals = usePoll(() => api.seals(), [sealRefresh]);

  const sealId: string | null = run?.seal_id ?? null;
  const stoppedAtStep: number | null = run?.stopped_at_step ?? null;
  const steps: any[] = run?.steps ?? [];
  const shown = stoppedAtStep === 1 ? steps.slice(0, 1) : steps;
  const compat = steps.find((s) => s.step === 2)?.compatibility ?? null;

  const doRun = async () => {
    if (!file || busy) return;
    setBusy(true);
    setRunErr(null);
    setRun(null);
    setReveal(null);
    setRevealErr(null);
    try {
      setRun(await api.validationRun(file));
      setSealRefresh((r) => r + 1);
    } catch (e: any) {
      setRunErr(e.message ?? "validation run failed");
    } finally {
      setBusy(false);
    }
  };

  const doReveal = async () => {
    if (!sealId || !labelFile || busy) return;
    setBusy(true);
    setRevealErr(null);
    try {
      setReveal(await api.validationReveal(sealId, labelFile, labelColumn.trim() || undefined));
      setSealRefresh((r) => r + 1);
    } catch (e: any) {
      setRevealErr(e.message ?? "reveal failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h2 className="page-title">Validation Lab</h2>
      <p className="page-sub">
        Upload a hidden validation file. The three steps run in a fixed order and
        the predictions are hashed before any label is read.
      </p>

      <div className="card">
        <h3>Step 0 · upload the file to be validated</h3>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <input type="file" accept=".csv,.xlsx"
                 onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <button className="btn" disabled={!file || busy} onClick={doRun}>
            {busy && !reveal ? "Running…" : "Run sealed validation"}
          </button>
          <span style={{ fontSize: 12, color: "var(--muted)" }}>
            .csv or .xlsx · a label column may be present; it is withheld before scoring
          </span>
        </div>
        {runErr && <ErrorState msg={runErr} />}
      </div>

      {run?.protocol && (
        <div className="notice" style={{ marginTop: 14 }}>
          <b>Protocol (verbatim from the API):</b>
          <div style={{ marginTop: 4 }}>{run.protocol}</div>
        </div>
      )}

      {run && (
        <div className="grid cols-4" style={{ marginTop: 14 }}>
          <div className="card">
            <h3>Run</h3>
            <div className="stat" style={{ fontSize: 15 }} title={run.run_id}>{run.run_id}</div>
            <div className="stat-sub">overall verdict <Verdict v={run.overall} /></div>
          </div>
          <div className="card">
            <h3>Compatibility score</h3>
            <div className="stat">
              {run.compatibility_score == null ? "–" : `${run.compatibility_score} / 100`}
            </div>
            <div className="stat-sub">{run.compatibility_band ?? "not computed — run stopped early"}</div>
          </div>
          <div className="card">
            <h3>Seal</h3>
            <div className="stat" style={{ fontSize: 13 }}>
              {sealId ? <span className="mono">{sealId}</span> : <span className="badge danger">no seal</span>}
            </div>
            <div className="stat-sub">{run.ui_state ?? "run stopped before predictions"}</div>
          </div>
          <div className="card">
            <h3>Next action (from API)</h3>
            <div className="stat" style={{ fontSize: 13 }}>{run.next_action ?? run.summary ?? "–"}</div>
            <div className="stat-sub">
              labels present in upload: {run.labels_available_for_reveal == null
                ? "–" : run.labels_available_for_reveal ? "yes" : "no"}
              {run.target_column_detected ? ` (${run.target_column_detected})` : ""}
            </div>
          </div>
        </div>
      )}

      {stoppedAtStep === 1 && (
        <div className="error-state" style={{ textAlign: "left", marginTop: 14 }}>
          <b>Run stopped at step 1. Steps 2 and 3 were never executed.</b>
          <div style={{ marginTop: 6 }}>{run.summary}</div>
          <div style={{ marginTop: 6, fontSize: 12 }}>
            No predictions were produced, so there is nothing to seal and nothing
            to reveal. Missing required features are refused rather than filled in.
          </div>
        </div>
      )}

      {shown.map((s) => <StepCard key={s.step} step={s} />)}

      {compat && <CompatibilityPanel compat={compat} />}

      <RevealPanel
        sealId={sealId}
        reveal={reveal}
        revealErr={revealErr}
        busy={busy}
        labelFile={labelFile}
        setLabelFile={setLabelFile}
        labelColumn={labelColumn}
        setLabelColumn={setLabelColumn}
        onReveal={doReveal}
      />

      <SealRegistry seals={seals} />

      <div className="footer-note">
        Uploaded rows are never used to retrain, recalibrate or re-threshold the
        model. A low compatibility score is printed next to the results; it never
        changes a prediction.
      </div>
    </>
  );
}

function StepCard({ step }: { step: any }) {
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <div className="step-head">
        <span className="step-num">{step.step}</span>
        <span className="step-title">{step.name}</span>
        <Verdict v={step.verdict} />
      </div>
      <p style={{ margin: "0 0 10px", fontSize: 13 }}>{step.status_detail}</p>
      {step.step === 1 && <Step1 s={step} />}
      {step.step === 2 && <Step2 s={step} />}
      {step.step === 3 && <Step3 s={step} />}
    </div>
  );
}

function Step1({ s }: { s: any }) {
  return (
    <div className="grid cols-2">
      <div>
        <table className="kv">
          <tbody>
            <tr><td>Rows in file</td><td>{s.n_rows?.toLocaleString()}</td></tr>
            <tr><td>Columns in file</td><td>{s.n_columns?.toLocaleString()}</td></tr>
            <tr><td>Required features</td><td>{s.n_required}</td></tr>
            <tr><td>Present</td><td>{s.n_present}</td></tr>
            <tr><td>Supplied by the file</td><td>{s.n_supplied_by_file}</td></tr>
            <tr><td>Derived here (MG_* block)</td><td>{s.n_derived_required}</td></tr>
            <tr><td>Schema completeness</td><td>{fmtPct(s.schema_completeness, 2)}</td></tr>
            <tr><td>Missing required</td><td>{s.n_missing_required}</td></tr>
            <tr><td>Unexpected columns (ignored)</td><td>{s.n_unexpected_columns}</td></tr>
            <tr><td>Duplicate rows</td><td>{s.duplicate_rows}</td></tr>
            <tr><td>Target column present in upload</td><td>{s.target_column_present ? "yes" : "no"}</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        {s.n_missing_required > 0 && (
          <>
            <h3>Missing required features</h3>
            <div className="mono">{(s.missing_required ?? []).join(", ")}</div>
          </>
        )}
        {(s.all_null_required ?? []).length > 0 && (
          <>
            <h3>Required features that are entirely empty</h3>
            <div className="mono">{s.all_null_required.join(", ")}</div>
          </>
        )}
        {(s.unparseable_numeric ?? []).length > 0 && (
          <>
            <h3>Columns that would not parse as numbers</h3>
            <div className="mono">{s.unparseable_numeric.join(", ")}</div>
          </>
        )}
        {(s.derived_required ?? []).length > 0 && (
          <>
            <h3>Derived required features</h3>
            <div className="mono">{s.derived_required.join(", ")}</div>
          </>
        )}
        {s.derivation_note && <div className="sealed-note">{s.derivation_note}</div>}
        <div className="sealed-note"><b>Guarantee:</b> {s.guarantee}</div>
      </div>
    </div>
  );
}

function Step2({ s }: { s: any }) {
  const a = s.adversarial ?? {};
  const shifted: any[] = a.top_shifted_features ?? [];
  return (
    <>
      <div className="grid cols-4">
        <div className="card">
          <h3>Adversarial AUC</h3>
          <div className="stat">{fmtNum(a.adversarial_auc, 4)}</div>
          <div className="stat-sub">0.50 = upload indistinguishable from training</div>
        </div>
        <div className="card">
          <h3>Shift band</h3>
          <div className="stat" style={{ fontSize: 15 }}>{a.shift_band ?? "–"}</div>
          <div className="stat-sub">
            {Object.entries(a.interpretation_scale ?? {}).map(([k, v]) => `${k} → ${v}`).join(" · ")}
          </div>
        </div>
        <div className="card">
          <h3>Missingness consistency</h3>
          <div className="stat">{fmtNum(s.missingness_consistency, 4)}</div>
          <div className="stat-sub">agreement of per-feature missing rates</div>
        </div>
        <div className="card">
          <h3>Value-range coverage</h3>
          <div className="stat">{fmtNum(s.value_range_coverage, 4)}</div>
          <div className="stat-sub">share of features whose upload median sits inside training p1–p99</div>
        </div>
      </div>
      <p style={{ fontSize: 13, marginTop: 12 }}>{a.verdict}</p>
      <div style={{ fontSize: 12, color: "var(--muted)" }}>
        compared {a.n_features_compared ?? "–"} features · {a.n_train?.toLocaleString() ?? "–"} training rows
        vs {a.n_upload?.toLocaleString() ?? "–"} uploaded rows
      </div>
      {shifted.length > 0 && (
        <div style={{ marginTop: 12, overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Feature</th><th>Name</th><th>|coef|</th><th>PSI</th>
                <th>Train median</th><th>Upload median</th>
                <th>Train missing</th><th>Upload missing</th>
              </tr>
            </thead>
            <tbody>
              {shifted.slice(0, 10).map((f) => (
                <tr key={f.feature}>
                  <td className="mono">{f.feature}</td>
                  <td>{f.variable_name ?? "—"}</td>
                  <td>{fmtNum(f.abs_coefficient, 3)}</td>
                  <td>{fmtNum(f.psi, 3)}</td>
                  <td>{fmtNum(f.train_median, 2)}</td>
                  <td>{fmtNum(f.upload_median, 2)}</td>
                  <td>{fmtPct(f.train_missing_rate, 2)}</td>
                  <td>{fmtPct(f.upload_missing_rate, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <ul style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 10, paddingLeft: 18 }}>
        {(s.guarantees ?? []).concat(a.guarantees ?? []).map((g: string, i: number) => (
          <li key={i}>{g}</li>
        ))}
      </ul>
    </>
  );
}

function Step3({ s }: { s: any }) {
  const seal = s.seal ?? {};
  return (
    <div className="grid cols-2">
      <div>
        <table className="kv">
          <tbody>
            <tr><td>Seal id</td><td className="mono">{seal.seal_id}</td></tr>
            <tr><td>Sealed (UTC)</td><td>{String(seal.sealed_utc ?? "").slice(0, 19)}</td></tr>
            <tr><td>Model version</td><td>{seal.model_version}</td></tr>
            <tr><td>Rows sealed</td><td>{seal.n_rows?.toLocaleString()}</td></tr>
            <tr><td>Score column</td><td className="mono">{seal.score_column}</td></tr>
            <tr><td>State</td><td><span className="badge info">{seal.state}</span></td></tr>
            <tr><td>Input SHA-256</td><td className="mono">{seal.input_sha256}</td></tr>
            <tr><td>Prediction SHA-256</td><td className="mono">{seal.prediction_sha256}</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        <div className="no-metric">
          <b>No metric has been computed at this point.</b>
          The predictions exist and are hashed. Nothing in this response has seen
          a label, so there is no accuracy, no PR-AUC and no recall to display yet.
        </div>
        <ul style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 10, paddingLeft: 18 }}>
          {(seal.notes ?? []).map((n: string, i: number) => <li key={i}>{n}</li>)}
        </ul>
      </div>
    </div>
  );
}

function CompatibilityPanel({ compat }: { compat: any }) {
  const weights: Record<string, number> = compat.weights ?? {};
  const components: Record<string, number> = compat.components ?? {};
  const contributions: Record<string, number> = compat.contributions ?? {};
  const keys = Object.keys(weights);
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <h3>Validation Compatibility Score — components and weights (from the API)</h3>
      <div className="grid cols-2">
        <div>
          <div className="stat">{compat.score} / 100</div>
          <div className="stat-sub">
            band <b>{compat.band}</b> — {compat.meaning}
          </div>
          <div className="sealed-note" style={{ marginTop: 12 }}>{compat.note}</div>
        </div>
        <div>
          <table className="data">
            <thead>
              <tr><th>Component</th><th>Value 0–1</th><th>Weight</th><th>Contribution</th></tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k}>
                  <td>
                    {k.replace(/_/g, " ")}
                    <div className="meter" style={{ marginTop: 4 }}>
                      <span style={{ width: `${Math.max(0, Math.min(1, components[k] ?? 0)) * 100}%` }} />
                    </div>
                  </td>
                  <td>{fmtNum(components[k], 4)}</td>
                  <td>{weights[k]}</td>
                  <td><b>{fmtNum(contributions[k], 2)}</b></td>
                </tr>
              ))}
              <tr>
                <td><b>Total</b></td><td>—</td>
                <td><b>{keys.reduce((t, k) => t + (weights[k] ?? 0), 0)}</b></td>
                <td><b>{compat.score}</b></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RevealPanel(props: {
  sealId: string | null;
  reveal: any;
  revealErr: string | null;
  busy: boolean;
  labelFile: File | null;
  setLabelFile: (f: File | null) => void;
  labelColumn: string;
  setLabelColumn: (s: string) => void;
  onReveal: () => void;
}) {
  const { sealId, reveal, revealErr, busy, labelFile, setLabelFile,
          labelColumn, setLabelColumn, onReveal } = props;
  const m = reveal?.metrics;
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <h3>Reveal — a separate, deliberate action</h3>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <input type="file" accept=".csv,.xlsx" disabled={!sealId}
               onChange={(e) => setLabelFile(e.target.files?.[0] ?? null)} />
        <input placeholder="label column (optional)" value={labelColumn}
               disabled={!sealId} style={{ width: 190 }}
               onChange={(e) => setLabelColumn(e.target.value)} />
        <button className="btn" disabled={!sealId || !labelFile || busy} onClick={onReveal}>
          Reveal Validation Metrics
        </button>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>
          {sealId
            ? `enabled: seal ${sealId} exists. Upload the labelled file — same rows, same order.`
            : "disabled until a sealed run exists. Nothing can be revealed before predictions are sealed."}
        </span>
      </div>

      {revealErr && <ErrorState msg={revealErr} />}

      {!reveal ? (
        <div className="no-metric">
          <b>No validation metric exists yet.</b>
          This panel is empty on purpose. Until the button above is pressed, no
          label has been read and the system has no accuracy figure of any kind
          for this file — not a hidden one, not a provisional one, none.
        </div>
      ) : (
        <div style={{ marginTop: 12 }}>
          <div className="sealed-note">
            <b>
              {reveal.verification?.verified
                ? "Seal verified before any label was read."
                : "Seal verification did not pass."}
            </b>{" "}
            {reveal.verification?.detail}
            <div style={{ marginTop: 6 }}>{reveal.integrity_statement}</div>
          </div>

          {m?.status === "OK" ? (
            <>
              <div className="grid cols-4" style={{ marginTop: 12 }}>
                <div className="card">
                  <h3>PR-AUC · revealed</h3>
                  <div className="stat">{fmtNum(m.pr_auc)}</div>
                  <div className="stat-sub">no-skill = prevalence {fmtNum(m.prevalence, 4)}</div>
                </div>
                <div className="card">
                  <h3>ROC-AUC</h3>
                  <div className="stat">{fmtNum(m.roc_auc)}</div>
                  <div className="stat-sub">secondary metric</div>
                </div>
                <div className="card">
                  <h3>Lift over prevalence</h3>
                  <div className="stat">{fmtNum(m.lift_over_prevalence, 2)}×</div>
                  <div className="stat-sub">PR-AUC divided by prevalence</div>
                </div>
                <div className="card">
                  <h3>Rows scored</h3>
                  <div className="stat">{m.n_rows_scored?.toLocaleString()}</div>
                  <div className="stat-sub">{m.n_positives} labelled mules in the revealed file</div>
                </div>
              </div>
              <table className="data" style={{ marginTop: 12 }}>
                <thead><tr><th>Review budget</th><th>Recall at budget</th><th>Precision at budget</th></tr></thead>
                <tbody>
                  {Object.keys(m.recall_at_budget ?? {}).map((k) => (
                    <tr key={k}>
                      <td>top {k}</td>
                      <td>{fmtPct(m.recall_at_budget[k])}</td>
                      <td>{fmtPct(m.precision_at_budget?.[k])}</td>
                    </tr>
                  ))}
                  <tr>
                    <td>MCC at top 100</td>
                    <td colSpan={2}>{fmtNum(m.mcc_at_top_100, 4)}</td>
                  </tr>
                </tbody>
              </table>
            </>
          ) : (
            <Empty msg={`${m?.status ?? "no metrics"} — ${m?.note ?? "the API returned no scorable metrics for this reveal."}`} />
          )}
        </div>
      )}
    </div>
  );
}

function SealRegistry({ seals }: { seals: ReturnType<typeof usePoll<any>> }) {
  const [open, setOpen] = useState<string | null>(null);
  const detail = usePoll(() => (open ? api.seal(open) : Promise.resolve(null as any)), [open]);
  const rows: any[] = seals.data?.seals ?? [];

  return (
    <div className="card" style={{ marginTop: 14 }}>
      <h3>Seal registry — every sealed run on this machine</h3>
      {seals.loading ? <Loading what="seal registry" /> :
       seals.error ? <ErrorState msg={seals.error} /> :
       rows.length === 0 ? <Empty msg="No sealed runs recorded yet." /> : (
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Seal</th><th>Sealed (UTC)</th><th>Model</th><th>Rows</th>
                <th>State</th><th>Prediction SHA-256</th><th>Verify</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 25).map((s) => (
                <tr key={s.seal_id}>
                  <td className="mono">{s.seal_id}</td>
                  <td style={{ fontSize: 12 }}>{String(s.sealed_utc ?? "").slice(0, 19)}</td>
                  <td>{s.model_version}</td>
                  <td>{s.n_rows}</td>
                  <td>
                    <span className={`badge ${s.state === "SEAL_BROKEN" ? "danger"
                      : s.state === "METRICS_REVEALED" ? "info" : "ok"}`}>
                      {s.state}
                    </span>
                  </td>
                  <td className="mono">{String(s.prediction_sha256 ?? "").slice(0, 16)}…</td>
                  <td>
                    <button className="btn secondary" onClick={() => setOpen(s.seal_id)}>
                      Re-hash now
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {open && (
        <div style={{ marginTop: 12 }}>
          {detail.loading ? <Loading what={`live verification of ${open}`} /> :
           detail.error ? <ErrorState msg={detail.error} /> :
           detail.data ? (
            <div className="sealed-note">
              <b>
                {detail.data.verification?.verified
                  ? "Hash matches the manifest."
                  : "Hash does NOT match the manifest."}
              </b>{" "}
              {detail.data.verification?.detail}
              <div className="mono" style={{ marginTop: 6 }}>
                {detail.data.seal?.seal_id} · sealed {String(detail.data.seal?.sealed_utc ?? "").slice(0, 19)} ·
                {" "}{detail.data.seal?.prediction_sha256}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
