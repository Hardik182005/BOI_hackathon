// Trinetra Mule-Farm Cohort Radar - the panel an analyst actually reads.
//
// Two things this component is careful about. It never draws an arrow between
// accounts, because an arrow means a transaction and the dataset has none. And
// it never renders a similarity without its percentile, because 0.88 means
// nothing until you know that random account pairs sit near 0.64.
//
// The disclaimer is not optional and not collapsible. It is the sentence that
// stops a prioritisation signal from being read as a finding about a person.
import { useState } from "react";
import { api, fmtNum } from "./api";
import { Empty, Loading } from "./components";

type Props = { caseId?: string; rowIndex?: number; title?: string };

const BAND_LABEL: Record<string, string> = {
  VERY_HIGH_SIMILARITY: "very high",
  HIGH_SIMILARITY: "high",
  MODERATE_SIMILARITY: "moderate",
  TYPICAL_SIMILARITY: "typical",
};

export default function CohortPanel({ caseId, rowIndex, title }: Props) {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [k, setK] = useState(10);
  const [open, setOpen] = useState<number | null>(null);

  const find = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = caseId
        ? await api.cohortForCase(caseId, k)
        : await api.cohortSearch({ row_index: rowIndex, k });
      setData(res);
    } catch (e: any) {
      setErr(e.message ?? String(e));
      setData(null);
    } finally {
      setBusy(false);
    }
  };

  const neighbors: any[] = data?.neighbors ?? [];
  const mutual = new Set<string>((data?.mutual_edges ?? []).map((e: any) => e.target));

  return (
    <div className="card">
      <h3>{title ?? "Behavioural cohort"}</h3>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button className="btn" disabled={busy} onClick={find}>
          {busy ? "Searching…" : "Find Behavioural Cohort"}
        </button>
        <label style={{ fontSize: 13 }}>
          neighbours{" "}
          <input type="number" min={1} max={25} value={k}
                 onChange={(e) => setK(Math.max(1, Math.min(25, Number(e.target.value) || 10)))}
                 style={{ width: 60 }} />
        </label>
      </div>

      {err && (
        <div className="notice" style={{ marginTop: 10 }}>
          Cohort unavailable — {err}
        </div>
      )}

      {busy && !data && <Loading what="behaviourally similar accounts" />}

      {data && (
        <>
          <div className="stat-sub" style={{ marginTop: 10 }}>
            Query {data.query_account} · risk {fmtNum(data.risk_probability)} ·{" "}
            {data.risk_tier} · reference: {data.reference_scope}
          </div>

          {neighbors.length === 0 ? (
            <Empty msg="No comparable accounts in the reference partition." />
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="data" style={{ marginTop: 10 }}>
                <thead>
                  <tr>
                    <th>Account</th>
                    <th>Similarity</th>
                    <th>Percentile</th>
                    <th>Band</th>
                    <th>Current risk</th>
                    <th>Shared patterns</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {neighbors.map((n: any, i: number) => (
                    <>
                      <tr key={n.account_reference}>
                        <td>
                          {n.account_reference}
                          {mutual.has(n.account_reference) && (
                            <span className="badge" style={{ marginLeft: 6 }}
                                  title="each account is among the other's nearest neighbours">
                              mutual
                            </span>
                          )}
                        </td>
                        <td><b>{fmtNum(n.behavioral_similarity, 4)}</b></td>
                        <td>
                          {n.beyond_empirical_null
                            ? "beyond sampled null"
                            : `${fmtNum(n.similarity_percentile, 2)}th`}
                        </td>
                        <td>
                          <span className={`badge ${
                            n.similarity_band === "VERY_HIGH_SIMILARITY" ? "danger" : ""}`}>
                            {BAND_LABEL[n.similarity_band] ?? n.similarity_band}
                          </span>
                        </td>
                        <td>
                          {fmtNum(n.neighbor_risk_probability)}{" "}
                          <span className="stat-sub">{n.neighbor_risk_tier}</span>
                        </td>
                        <td style={{ fontSize: 12 }}>
                          {(n.shared_patterns ?? []).join(", ") || "—"}
                        </td>
                        <td>
                          <button className="btn secondary" style={{ padding: "2px 8px" }}
                                  onClick={() => setOpen(open === i ? null : i)}>
                            {open === i ? "hide" : "why"}
                          </button>
                        </td>
                      </tr>
                      {open === i && (
                        <tr key={`${n.account_reference}-why`}>
                          <td colSpan={7} style={{ background: "var(--panel-2, #fafafa)" }}>
                            <div className="grid cols-2" style={{ gap: 12 }}>
                              <div>
                                <b style={{ fontSize: 12 }}>Why similar</b>
                                <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 12 }}>
                                  {(n.main_shared_features ?? []).map((f: any) => (
                                    <li key={f.feature}>
                                      {f.description || f.feature_name}
                                      <span className="stat-sub"> ({f.feature})</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <b style={{ fontSize: 12 }}>Where they differ</b>
                                <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 12 }}>
                                  {(n.main_differences ?? []).map((f: any) => (
                                    <li key={f.feature}>
                                      {f.description || f.feature_name}
                                      <span className="stat-sub"> ({f.feature})</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                            <div className="stat-sub" style={{ marginTop: 8 }}>
                              Pattern agreement (Jaccard over typology cards):{" "}
                              {fmtNum(n.pattern_similarity, 3)} — reported separately
                              from behavioural similarity, never blended into it.
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {data.cohort_summary && (
            <div className="stat-sub" style={{ marginTop: 8 }}>
              {data.cohort_summary.n_neighbors} neighbours ·{" "}
              {data.cohort_summary.high_risk_neighbors} already at a review tier ·
              median neighbour risk {fmtNum(data.cohort_summary.median_neighbor_risk)}
            </div>
          )}

          <div className="notice" style={{ marginTop: 10 }}>
            <b>{data.disclaimer}</b>
          </div>
          <div className="stat-sub" style={{ marginTop: 6 }}>
            {data.action_policy}
          </div>
        </>
      )}
    </div>
  );
}
