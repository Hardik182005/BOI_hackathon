import { useEffect, useState } from "react";
import { api, fmtNum } from "../api";
import { Empty, ErrorState, Loading, TierBadge, usePoll } from "../components";
import CohortPanel from "../CohortPanel";

// Graph Lab. The transaction-graph adapter takes an OPTIONAL edge file. This
// page does not assume one exists: it asks the running backend which routes it
// actually serves (/openapi.json) and reports the answer. When no counterparty
// ingestion route is deployed and no edge file has been supplied, the page says
// so and draws nothing. Deriving sender/receiver pairs from aggregate F-columns
// would be fabrication, and the system refuses to do it.

const EDGE_REQUIREMENTS = ["sender", "receiver", "amount", "timestamp"];

export default function GraphLab() {
  const [tab, setTab] = useState<"evidence" | "transaction" | "cohort">("transaction");
  const spec = usePoll(() => api.openapi(), []);
  const cases = usePoll(() => api.cases("?limit=25"), []);

  const paths: string[] = spec.data?.paths ? Object.keys(spec.data.paths) : [];
  const graphRoutes = paths.filter((p) => p.startsWith("/v1/graph"));
  const proofRoutes = paths.filter((p) => p.startsWith("/v1/proofgraph"));

  return (
    <>
      <h2 className="page-title">Graph Lab</h2>
      <p className="page-sub">
        Two kinds of graph. One is built from evidence this pipeline actually
        produces. The other needs counterparty data that this dataset does not
        contain.
      </p>

      <div className="tabs">
        <button className={tab === "transaction" ? "active" : ""}
                onClick={() => setTab("transaction")}>
          Transaction Graph (requires edge file)
        </button>
        <button className={tab === "evidence" ? "active" : ""}
                onClick={() => setTab("evidence")}>
          ProofGraph (evidence graph)
        </button>
        <button className={tab === "cohort" ? "active" : ""}
                onClick={() => setTab("cohort")}>
          Behavioural Cohort Radar
        </button>
      </div>

      {tab === "transaction" ? (
        <TransactionTab spec={spec} graphRoutes={graphRoutes} />
      ) : tab === "cohort" ? (
        <CohortTab cases={cases} />
      ) : (
        <EvidenceTab cases={cases} proofRoutes={proofRoutes} />
      )}

      <div className="footer-note">
        No relationship, counterparty or transfer shown anywhere in MuleGuard is
        inferred. If the data does not contain it, the interface says the data
        does not contain it.
      </div>
    </>
  );
}

// The adapter's own answer, not an inference from the route list. A deployed
// route says the feature exists; only /v1/graph/status says whether a real edge
// file was ever supplied, and that distinction is the entire point of the page.
function useGraphState(available: boolean) {
  const [state, setState] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = () => {
    if (!available) return;
    api.graphStatus().then(setState).catch((e) => setErr(String(e.message ?? e)));
  };
  useEffect(refresh, [available]);

  const upload = (f: File) => {
    setBusy(true);
    setErr(null);
    api.graphUpload(f)
      .then(setState)
      .catch((e) => setErr(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };
  const discard = () => {
    api.graphDiscard().then(refresh).catch((e) => setErr(String(e.message ?? e)));
  };
  return { state, busy, err, upload, discard };
}

function LoadedGraph({ state, onDiscard }: { state: any; onDiscard: () => void }) {
  const f = state.edge_file;
  return (
    <>
      <div className="card">
        <div className="step-head">
          <span className="step-num">✓</span>
          <span className="step-title">A transaction graph is loaded</span>
          <span className="verdict PASS">FROM AN UPLOADED EDGE FILE</span>
        </div>
        <table className="kv">
          <tbody>
            <tr><td>Edges supplied</td><td>{fmtNum(f.n_edges_supplied, 0)}</td></tr>
            <tr><td>Edges used</td><td>{fmtNum(f.n_edges_used, 0)}</td></tr>
            <tr><td>Self-loops dropped</td><td>{fmtNum(f.n_self_loops_dropped, 0)}</td></tr>
            <tr>
              <td>Non-positive amounts dropped</td>
              <td>{fmtNum(f.n_non_positive_amounts_dropped, 0)}</td>
            </tr>
            <tr><td>Accounts</td><td>{fmtNum(f.n_accounts, 0)}</td></tr>
            <tr><td>Window</td><td className="mono">{f.window_start} → {f.window_end}</td></tr>
          </tbody>
        </table>
        <div className="notice">
          Loading this file changed <b>no</b> risk score. The model was fitted
          without any graph feature, so every account is scored identically
          whether this graph exists or not.
        </div>
        <button className="secondary" onClick={onDiscard} style={{ marginTop: 10 }}>
          Discard this graph
        </button>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Highest pass-through accounts</h3>
          <p style={{ fontSize: 13 }}>
            Value in that leaves again, measured by amount rather than count.
          </p>
          <table className="data">
            <thead>
              <tr><th>Account</th><th>Fan in</th><th>Fan out</th><th>Pass-through</th><th>Dwell (h)</th></tr>
            </thead>
            <tbody>
              {state.top_passthrough_accounts.slice(0, 10).map((r: any) => (
                <tr key={r.account}>
                  <td className="mono">{r.account}</td>
                  <td>{r.fan_in_counterparties}</td>
                  <td>{r.fan_out_counterparties}</td>
                  <td>{fmtNum(r.passthrough_ratio, 3)}</td>
                  <td>{r.median_dwell_hours ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h3>What this graph may and may not do</h3>
          <ul style={{ fontSize: 13, paddingLeft: 18 }}>
            {(state.contract ?? []).map((c: string) => <li key={c}>{c}</li>)}
          </ul>
        </div>
      </div>
    </>
  );
}

function TransactionTab({ spec, graphRoutes }: { spec: any; graphRoutes: string[] }) {
  const available = graphRoutes.length > 0;
  const graph = useGraphState(available);

  if (spec.loading) return <Loading what="the backend route list" />;
  if (spec.error) {
    return (
      <>
        <ErrorState msg={spec.error} />
        <div className="card" style={{ marginTop: 14 }}>
          <h3>What this page would need</h3>
          <p style={{ fontSize: 13 }}>
            The backend could not be reached, so this page cannot tell you
            whether a counterparty-edge route is deployed. It will not guess, and
            it will not draw a graph in the meantime.
          </p>
        </div>
      </>
    );
  }

  if (graph.state?.status === "OK") {
    return <LoadedGraph state={graph.state} onDiscard={graph.discard} />;
  }

  return (
    <>
      <div className="card">
        <div className="step-head">
          <span className="step-num">!</span>
          <span className="step-title">No transaction graph is available</span>
          <span className={`verdict ${available ? "WARN" : "FAIL"}`}>
            {available ? "ROUTE PRESENT · NO EDGE FILE LOADED" : "NO EDGE INGESTION ROUTE"}
          </span>
        </div>
        <p style={{ fontSize: 13.5, margin: "0 0 10px" }}>
          The supplied dataset is <b>aggregate per account</b>: every column is a
          summary statistic for one account over a window. It contains no
          counterparty identifiers, so there is no pair of accounts anywhere in
          it between which an edge could be drawn.
        </p>
        <p style={{ fontSize: 13.5, margin: "0 0 10px" }}>
          <b>
            MuleGuard refuses to fabricate a transaction graph from aggregate
            F-columns.
          </b>{" "}
          A network diagram built by pairing accounts that happen to have similar
          aggregate values would look convincing and would mean nothing. Nodes
          that a reviewer cannot verify are worse than no diagram at all, so this
          tab stays empty until real edges are supplied.
        </p>
        <div className="notice">
          The transaction-graph adapter accepts an <b>optional</b> edge file. It
          is not part of the scoring path: the model that produces every score in
          this system was fitted without any graph feature, so nothing on the
          rest of the dashboard depends on this tab.
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Minimum columns an edge file must contain</h3>
          <table className="data">
            <thead><tr><th>Column</th><th>Meaning</th></tr></thead>
            <tbody>
              <tr><td className="mono">sender</td><td>originating account reference (masked)</td></tr>
              <tr><td className="mono">receiver</td><td>beneficiary account reference (masked)</td></tr>
              <tr><td className="mono">amount</td><td>transferred value</td></tr>
              <tr><td className="mono">timestamp</td><td>when the transfer occurred</td></tr>
            </tbody>
          </table>
          <div className="notice">
            All four are required. With fewer than four, fan-in, fan-out,
            pass-through timing and cycle detection are all undefined, and a
            partial graph would invite exactly the false confidence this page
            exists to prevent.
          </div>
        </div>

        <div className="card">
          <h3>What the running backend actually serves</h3>
          {available ? (
            <>
              <p style={{ fontSize: 13 }}>
                A counterparty route is deployed on this backend, but no edge
                file has been ingested in this session, so there is still nothing
                to draw:
              </p>
              <ul className="mono" style={{ paddingLeft: 18 }}>
                {graphRoutes.map((p) => <li key={p}>{p}</li>)}
              </ul>
              <div style={{ marginTop: 10 }}>
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) graph.upload(f);
                  }}
                />
                {graph.busy && <p style={{ fontSize: 13 }}>Building the graph…</p>}
                {graph.err && <ErrorState msg={graph.err} />}
              </div>
              <div className="notice">
                Uploading an edge file adds corroborating structure to cases that
                already have a score. It does not re-score anything, and the file
                is held in memory for this session only — it is never written to
                disk.
              </div>
            </>
          ) : (
            <>
              <p style={{ fontSize: 13 }}>
                This build exposes <b>no</b> route beginning{" "}
                <span className="mono">/v1/graph</span>. Checked live against{" "}
                <span className="mono">/openapi.json</span> on the backend serving
                this page, not assumed:
              </p>
              <table className="kv">
                <tbody>
                  <tr><td>Edge upload route</td><td>not deployed</td></tr>
                  <tr><td>Edge file loaded</td><td>none</td></tr>
                  <tr>
                    <td>Required columns</td>
                    <td className="mono">{EDGE_REQUIREMENTS.join(", ")}</td>
                  </tr>
                  <tr><td>Graph features used by the model</td><td>none</td></tr>
                </tbody>
              </table>
            </>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>What would be computed if real edges were supplied</h3>
        <p style={{ fontSize: 13, color: "var(--muted)", margin: "0 0 8px" }}>
          Listed so the gap is explicit. None of these is being estimated,
          approximated or displayed anywhere in the product today.
        </p>
        <div className="grid cols-3">
          <ul style={{ fontSize: 13, margin: 0, paddingLeft: 18 }}>
            <li>strongly connected components</li>
            <li>directed cycles</li>
            <li>fan-in and fan-out degree</li>
          </ul>
          <ul style={{ fontSize: 13, margin: 0, paddingLeft: 18 }}>
            <li>rapid pass-through chains</li>
            <li>smurfing within a time window</li>
            <li>shell / intermediary chains</li>
          </ul>
          <ul style={{ fontSize: 13, margin: 0, paddingLeft: 18 }}>
            <li>unique sender and receiver counts</li>
            <li>transaction velocity</li>
            <li>active timespan</li>
          </ul>
        </div>
      </div>
    </>
  );
}

function EvidenceTab({ cases, proofRoutes }: { cases: any; proofRoutes: string[] }) {
  const rows: any[] = cases.data?.cases ?? [];
  return (
    <>
      <div className="card">
        <h3>Evidence graph — available for every scored case</h3>
        <p style={{ fontSize: 13.5, margin: "0 0 8px" }}>
          The ProofGraph is built from the case's own verified score payload:
          named feature evidence, model votes, uncertainty and the review
          recommendation. Each node carries the dataset column or pipeline metric
          it came from.
        </p>
        {proofRoutes.length > 0 && (
          <div className="notice">
            Served by:{" "}
            <span className="mono">{proofRoutes.join(", ")}</span>
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Open an evidence graph</h3>
        {cases.loading ? <Loading what="cases" /> :
         cases.error ? <ErrorState msg={cases.error} /> :
         rows.length === 0 ? (
          <Empty msg="No scored cases exist yet, so there is no evidence graph to open. Score accounts through the API or the batch upload to populate the queue." />
        ) : (
          <table className="data">
            <thead>
              <tr><th>Case</th><th>Account (masked)</th><th>Tier</th><th>Calibrated risk</th><th>Graph</th></tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.case_id}>
                  <td className="mono">{c.case_id}</td>
                  <td>{c.account_reference}</td>
                  <td><TierBadge tier={c.risk_tier} /></td>
                  <td>{fmtNum(c.calibrated_risk)}</td>
                  <td><a href={`#/proof/${c.case_id}`}>open ProofGraph</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}

// Behavioural Cohort Radar. The third graph on this page, and the only one of
// the three whose edges the dataset actually supports: not who paid whom, but
// which accounts behave unusually alike. The edge label says exactly that and
// nothing more.
function CohortTab({ cases }: { cases: any }) {
  const [selected, setSelected] = useState<string>("");
  const rows: any[] = cases.data?.cases ?? [];

  return (
    <>
      <div className="card">
        <h3>Pick a case</h3>
        <p className="stat-sub">
          The radar retrieves behaviourally similar accounts from a frozen
          development-only reference partition. It does not score, re-score or
          re-rank anything — the risk shown against each neighbour is the one
          the classifier already produced.
        </p>
        {cases.loading ? <Loading what="cases" /> : rows.length === 0 ? (
          <Empty msg="No cases yet — score an account first." />
        ) : (
          <select value={selected} onChange={(e) => setSelected(e.target.value)}
                  style={{ minWidth: 320 }}>
            <option value="">— select a case —</option>
            {rows.map((c: any) => (
              <option key={c.case_id} value={c.case_id}>
                {c.case_id} · {c.risk_tier} · {fmtNum(c.calibrated_risk)}
              </option>
            ))}
          </select>
        )}
      </div>

      {selected && (
        <div style={{ marginTop: 14 }}>
          <CohortPanel caseId={selected} title="Behaviourally similar accounts" />
        </div>
      )}
    </>
  );
}
