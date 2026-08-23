import { useState } from "react";
import { useParams } from "react-router-dom";
import { api, fmtNum } from "../api";
import { Empty, ErrorState, HumanReviewNotice, Loading, TierBadge, usePoll } from "../components";

// Dual-evidence ProofGraph. Everything on this page is a re-presentation of
// /v1/proofgraph/{case_id}: no node, edge, weight or argument is constructed
// here. The graph is drawn as plain SVG - no graph library, no CDN.

const NODE_STYLE: Record<string, { fill: string; stroke: string; label: string }> = {
  ACCOUNT: { fill: "#ffffff", stroke: "#111111", label: "Account" },
  EVIDENCE_FOR: { fill: "#fef2f2", stroke: "#dc2626", label: "Evidence raising risk" },
  EVIDENCE_AGAINST: { fill: "#eff6ff", stroke: "#2563eb", label: "Evidence against the alert" },
  MODEL_VOTE: { fill: "#f3f5f9", stroke: "#94a3b8", label: "Model vote" },
  UNCERTAINTY: { fill: "#fffbeb", stroke: "#d97706", label: "Uncertainty" },
  COUNTERFACTUAL: { fill: "#f5f3ff", stroke: "#7c3aed", label: "Counterfactual" },
  COUNTERFACTUAL_TWIN: { fill: "#f5f3ff", stroke: "#7c3aed", label: "Counterfactual twin" },
  PATTERN: { fill: "#f3f5f9", stroke: "#5b6675", label: "Behavioural pattern" },
  DECISION: { fill: "#ffffff", stroke: "#111111", label: "Review recommendation" },
  CONTROL_ATTRIBUTION: { fill: "#fafafa", stroke: "#5b6675",
                         label: "Limitation — requires human verification" },
};

// Addendum caps, so the canvas stays readable rather than becoming a spaghetti
// graph. "Show every node" lifts them.
const CAPS: Record<string, number> = {
  EVIDENCE_FOR: 5, EVIDENCE_AGAINST: 3, MODEL_VOTE: 3, UNCERTAINTY: 1,
};

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const byWeight = (a: any, b: any) => (b.weight ?? 0) - (a.weight ?? 0);

export default function ProofGraph() {
  const { caseId = "" } = useParams();
  const [showAll, setShowAll] = useState(false);
  const [saveErr, setSaveErr] = useState<string | null>(null);
  const { data, error, loading } = usePoll(() => api.proofgraph(caseId), [caseId]);

  // The file is the endpoint's response, byte for byte - not this component's
  // parsed copy of it re-serialised. A reviewer who keeps the JSON can hand it
  // to someone who re-requests the same case and get the same object back.
  const saveJson = async () => {
    setSaveErr(null);
    try {
      const text = await api.proofgraphJson(caseId);
      const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `proofgraph_${caseId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setSaveErr(e?.message ?? "the ProofGraph could not be downloaded");
    }
  };

  if (loading) return <Loading what={`evidence graph for ${caseId}`} />;
  if (error) return <ErrorState msg={error} />;
  if (!data) return <Empty msg="No evidence graph returned for this case." />;

  const nodes: any[] = data.nodes ?? [];
  const edges: any[] = data.edges ?? [];
  const court = data.courtroom ?? {};
  const counts = data.evidence_counts ?? {};
  const twin = nodes.find((n) => n.type === "COUNTERFACTUAL_TWIN");

  return (
    <>
      <h2 className="page-title">ProofGraph · {caseId}</h2>
      <p className="page-sub">
        Account {data.account_reference} · model {data.model_version ?? "–"} ·
        built {String(data.generated_utc ?? "").slice(0, 19)} ·{" "}
        <a href={`#/cases/${caseId}`}>back to the case file</a>
      </p>
      <HumanReviewNotice />

      <div className="grid cols-4">
        <div className="card">
          <h3>Review tier</h3>
          <div className="stat"><TierBadge tier={data.risk_tier ?? "MONITOR"} /></div>
          <div className="stat-sub">behavioural ranking for a human reviewer</div>
        </div>
        <div className="card">
          <h3>Calibrated risk</h3>
          <div className="stat">{fmtNum(data.calibrated_risk)}</div>
          <div className="stat-sub">calibrated probability, not a finding</div>
        </div>
        <div className="card">
          <h3>Evidence on the record</h3>
          <div className="stat" style={{ fontSize: 17 }}>
            {counts.prosecution ?? 0} for · {counts.defence ?? 0} against
          </div>
          <div className="stat-sub">{counts.uncertainty ?? 0} uncertainty node(s)</div>
        </div>
        <div className="card">
          <h3>Evidence balance</h3>
          <div className="stat">{fmtNum(court.evidence_balance)}</div>
          <div className="stat-sub">
            prosecution weight {fmtNum(court.prosecution_weight)} · defence weight {fmtNum(court.defence_weight)}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <h3 style={{ margin: 0 }}>Evidence graph — every node names the column or metric it came from</h3>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn secondary" onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Show the top nodes only" : "Show every node"}
            </button>
            <button className="btn secondary" onClick={saveJson}>
              Download ProofGraph JSON
            </button>
          </div>
        </div>
        {saveErr && <ErrorState msg={saveErr} />}
        <GraphCanvas nodes={nodes} edges={edges} showAll={showAll} />
        <div className="legend">
          {Object.entries(NODE_STYLE).map(([k, s]) => (
            <span key={k}>
              <i style={{ background: s.fill, borderColor: s.stroke }} />
              {s.label}
            </span>
          ))}
        </div>
        <div className="notice">{data.provenance_statement}</div>
        <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 6 }}>
          The download is the /v1/proofgraph/{caseId} response itself, so it holds
          every node the API returned — including the lower-weight ones this canvas
          caps — rather than the subset drawn above.
        </div>
      </div>

      <h3 style={{ fontSize: 14, margin: "20px 0 8px" }}>Model Courtroom</h3>
      <div className="court">
        <ArgumentSide
          title="Prosecution"
          subtitle="points that raise the risk"
          weight={court.prosecution_weight}
          items={court.prosecution ?? []}
          accent="#dc2626"
        />
        <ArgumentSide
          title="Defence"
          subtitle="points that weaken the alert"
          weight={court.defence_weight}
          items={court.defence ?? []}
          accent="#2563eb"
        />
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="side-head">
          <h4>Verdict</h4>
          <span className="badge info">{court.contested ? "contested" : "not contested"}</span>
        </div>
        <div className="stat" style={{ fontSize: 18 }}>
          {String(court.verdict ?? "–").replace(/_/g, " ")}
        </div>
        <p style={{ fontSize: 13, marginTop: 8 }}>{court.verdict_rationale}</p>
        <div className="notice">{court.evidence_policy}</div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Disagreement d={data.disagreement} />
        <Twin twin={twin} />
      </div>

      <div className="footer-note">
        A high evidence balance is a reason to review sooner, never a conclusion
        about the account holder. An account with no current flags is not
        certified — it is simply not currently flagged.
      </div>
    </>
  );
}

// ---------------------------------------------------------------- graph SVG

const W = 1120;
const COL_W = 344;
const NODE_H = 46;
const GAP = 10;
const TOP_Y = 108;
const LEFT_X = 8;
const RIGHT_X = W - COL_W - 8;
const MID_X = (W - COL_W) / 2;

type Placed = { node: any; x: number; y: number; w: number; h: number; anchor: "left" | "right" | "top" };

function GraphCanvas({ nodes, edges, showAll }: { nodes: any[]; edges: any[]; showAll: boolean }) {
  const pick = (type: string) => {
    const all = nodes.filter((n) => n.type === type).sort(byWeight);
    return showAll || CAPS[type] == null ? all : all.slice(0, CAPS[type]);
  };

  const account = nodes.find((n) => n.type === "ACCOUNT");
  const forN = pick("EVIDENCE_FOR");
  const againstN = pick("EVIDENCE_AGAINST");
  const uncertain = pick("UNCERTAINTY");
  const votes = pick("MODEL_VOTE");
  const others = nodes.filter((n) =>
    ["PATTERN", "COUNTERFACTUAL", "COUNTERFACTUAL_TWIN"].includes(n.type));
  const decision = nodes.find((n) => n.type === "DECISION");
  const control = nodes.find((n) => n.type === "CONTROL_ATTRIBUTION");

  const placed: Placed[] = [];
  const stack = (list: any[], x: number, anchor: Placed["anchor"], startY: number) => {
    let y = startY;
    for (const node of list) {
      placed.push({ node, x, y, w: COL_W, h: NODE_H, anchor });
      y += NODE_H + GAP;
    }
    return y;
  };

  const leftEnd = stack(forN, LEFT_X, "right", TOP_Y);
  const rightEnd = stack(againstN.concat(uncertain), RIGHT_X, "left", TOP_Y);
  let midEnd = stack(votes.concat(others), MID_X, "top", TOP_Y);
  if (decision) {
    midEnd = Math.max(midEnd, Math.max(leftEnd, rightEnd)) + 14;
    placed.push({ node: decision, x: MID_X, y: midEnd, w: COL_W, h: 52, anchor: "top" });
    midEnd += 52;
  }
  // Section 22: the limitation hangs off the decision, not off the account.
  // Drawn last and below, because it qualifies what may be done with the
  // recommendation rather than contributing to it.
  if (control) {
    midEnd += 14;
    placed.push({ node: control, x: MID_X, y: midEnd, w: COL_W, h: 52, anchor: "top" });
    midEnd += 52;
  }

  const height = Math.max(leftEnd, rightEnd, midEnd) + 16;
  const accountBox = { x: (W - 360) / 2, y: 12, w: 360, h: 56 };
  const pos = new Map(placed.map((p) => [p.node.id, p]));
  const hidden = nodes.length - placed.length - (account ? 1 : 0);

  if (!account && placed.length === 0) {
    return <Empty msg="The graph returned no nodes to draw." />;
  }

  const anchorPoint = (p: Placed) =>
    p.anchor === "right" ? { x: p.x + p.w, y: p.y + p.h / 2 }
    : p.anchor === "left" ? { x: p.x, y: p.y + p.h / 2 }
    : { x: p.x + p.w / 2, y: p.y };

  return (
    <>
      <div className="graph-canvas" style={{ marginTop: 10 }}>
        <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={Math.min(height, 620)}
             role="img" aria-label="evidence graph" style={{ minWidth: 720, display: "block" }}>
          {edges.map((e, i) => {
            const to = pos.get(e.target);
            if (!to) return null;
            // The one edge that does not originate at the account: the
            // limitation attached to the decision (REQUIRES_HUMAN_VERIFICATION).
            if (e.source === "decision" && to.node.type === "CONTROL_ATTRIBUTION") {
              const from = pos.get("decision");
              if (!from) return null;
              return (
                <line key={`ctl-${i}`}
                      x1={from.x + from.w / 2} y1={from.y + from.h}
                      x2={to.x + to.w / 2} y2={to.y}
                      stroke="#5b6675" strokeWidth={1.2} strokeDasharray="4 3" />
              );
            }
            if (e.source !== "account") return null;
            const a = anchorPoint(to);
            const from = { x: accountBox.x + accountBox.w / 2, y: accountBox.y + accountBox.h };
            const stroke = NODE_STYLE[to.node.type]?.stroke ?? "#94a3b8";
            return (
              <g key={`${e.source}-${e.target}-${i}`}>
                <path d={`M ${from.x} ${from.y} C ${from.x} ${(from.y + a.y) / 2}, ${a.x} ${(from.y + a.y) / 2}, ${a.x} ${a.y}`}
                      fill="none" stroke={stroke} strokeWidth={1.1} opacity={0.55} />
              </g>
            );
          })}

          {account && (
            <Box x={accountBox.x} y={accountBox.y} w={accountBox.w} h={accountBox.h}
                 node={account} big />
          )}
          {placed.map((p) => (
            <Box key={p.node.id} x={p.x} y={p.y} w={p.w} h={p.h} node={p.node} />
          ))}
        </svg>
      </div>
      <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 6 }}>
        {placed.length + (account ? 1 : 0)} of {nodes.length} nodes drawn
        {hidden > 0 && !showAll ? ` · ${hidden} lower-weight node(s) hidden — use “Show every node”` : ""}
        {" · "}edges shown: relations from the account node
        {control ? ", plus the decision's human-verification limitation" : ""}
      </div>
    </>
  );
}

function Box({ x, y, w, h, node, big }: {
  x: number; y: number; w: number; h: number; node: any; big?: boolean;
}) {
  const s = NODE_STYLE[node.type] ?? { fill: "#ffffff", stroke: "#5b6675", label: node.type };
  const val = node.value == null ? ""
    : typeof node.value === "number" ? node.value.toFixed(4) : String(node.value);
  return (
    <g>
      <title>{`${node.label}\nsource: ${node.source}\n${node.detail ?? ""}`}</title>
      <rect x={x} y={y} width={w} height={h} rx={7} fill={s.fill} stroke={s.stroke}
            strokeWidth={big ? 1.8 : 1.1} />
      <text x={x + 12} y={y + (big ? 24 : 19)} fontSize={big ? 13.5 : 12.5}
            fontWeight={600} fill="#111111">
        {clip(String(node.label ?? node.id), big ? 46 : 40)}
      </text>
      <text x={x + 12} y={y + (big ? 42 : 35)} fontSize={11} fill="#5b6675">
        {clip(`${node.source}${val ? ` · ${val}` : ""}${node.weight ? ` · w ${Number(node.weight).toFixed(3)}` : ""}`, big ? 56 : 50)}
      </text>
    </g>
  );
}

// ------------------------------------------------------------- courtroom UI

function ArgumentSide({ title, subtitle, weight, items, accent }: {
  title: string; subtitle: string; weight: number | undefined;
  items: any[]; accent: string;
}) {
  return (
    <div className="card" style={{ borderTop: `3px solid ${accent}` }}>
      <div className="side-head">
        <h4>{title}</h4>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>
          total weight {fmtNum(weight)}
        </span>
      </div>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>{subtitle}</div>
      {items.length === 0 ? (
        <Empty msg={`The graph contains no ${title.toLowerCase()} points for this case.`} />
      ) : items.map((a, i) => (
        <div className="arg" key={i}>
          <div className="pt">{a.point}</div>
          <div className="why">{a.because}</div>
          <div className="src">
            source <span className="mono">{a.source}</span> · weight {fmtNum(a.weight, 4)}
          </div>
        </div>
      ))}
    </div>
  );
}

function Disagreement({ d }: { d: any }) {
  const families: Record<string, number> = d?.families ?? {};
  const names = Object.keys(families);
  return (
    <div className="card">
      <h3>Model disagreement</h3>
      {!d || names.length === 0 ? (
        <Empty msg="No per-family scores were stored for this case." />
      ) : (
        <>
          <div className="stat" style={{ fontSize: 16 }}>
            <span className={`badge ${d.status === "MODEL_CONSENSUS" ? "ok"
              : d.status === "PARTIAL_AGREEMENT" ? "info" : "warn"}`}>
              {String(d.status).replace(/_/g, " ")}
            </span>
          </div>
          <table className="data" style={{ marginTop: 10 }}>
            <thead><tr><th>Model family</th><th>Probability</th></tr></thead>
            <tbody>
              {names.map((n) => (
                <tr key={n}><td>{n}</td><td><b>{fmtNum(families[n], 4)}</b></td></tr>
              ))}
            </tbody>
          </table>
          <table className="kv" style={{ marginTop: 10 }}>
            <tbody>
              <tr><td>mean</td><td>{fmtNum(d.mean, 4)}</td></tr>
              <tr><td>std</td><td>{fmtNum(d.std, 4)}</td></tr>
              <tr><td>rank std</td><td>{fmtNum(d.rank_std, 4)}</td></tr>
              <tr><td>max − min</td><td>{fmtNum(d.max_minus_min, 4)}</td></tr>
            </tbody>
          </table>
          <div className="notice">{d.interpretation}</div>
        </>
      )}
    </div>
  );
}

function Twin({ twin }: { twin: any }) {
  const diffs: any[] = twin?.extra?.differences ?? [];
  return (
    <div className="card">
      <h3>Counterfactual twin — the closest account that was not escalated</h3>
      {!twin ? (
        <Empty msg="No counterfactual twin is available for this case. The twin index needs the development matrix; when it cannot be built, this panel stays empty rather than showing an invented neighbour." />
      ) : (
        <>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{twin.label}</div>
          <p style={{ fontSize: 12.5, color: "var(--muted)" }}>{twin.detail}</p>
          {diffs.length === 0 ? (
            <Empty msg="The twin matched on every deciding feature; no differences were returned." />
          ) : (
            <table className="data">
              <thead>
                <tr><th>Feature</th><th>Name</th><th>This account</th><th>Twin</th><th>Gap</th></tr>
              </thead>
              <tbody>
                {diffs.map((d) => (
                  <tr key={d.feature}>
                    <td className="mono">{d.feature}</td>
                    <td>{d.variable_name ?? "—"}</td>
                    <td>{fmtNum(d.this_account, 2)}</td>
                    <td>{fmtNum(d.twin_account, 2)}</td>
                    <td>{fmtNum(d.absolute_gap, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
