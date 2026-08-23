// Account-Control Ambiguity Guardrail (sections 20-21).
//
// Three questions, kept visually apart because they have different answers and
// different sources. The classifier answers the first one. Nothing in this
// project answers the other two, and the card says so in place of a guess.
//
// The checklist is what a person should go and obtain. It is not a set of
// lookups this system can perform - every row is marked with where it stands,
// so the card cannot be mistaken for evidence already held.
import { fmtNum } from "./api";

type Props = { card: any };

const BAND_CLASS: Record<string, string> = {
  HIGH: "danger",
  MODERATE: "",
  NOT_COMPARABLE: "",
  LOW: "ok",
};

export default function ControlAttribution({ card }: Props) {
  if (!card) return null;
  const risk = card.behavioural_mule_risk ?? {};
  const control = card.account_control_evidence ?? {};
  const intent = card.intent_attribution ?? {};

  return (
    <div className="card">
      <h3>Control attribution</h3>

      <table className="data">
        <tbody>
          <tr>
            <td style={{ width: "42%" }}>Behavioural mule risk</td>
            <td>
              <span className={`badge ${BAND_CLASS[risk.band] ?? ""}`}>{risk.band}</span>{" "}
              <span className="stat-sub">
                {fmtNum(risk.risk_probability)} · {risk.risk_tier}
              </span>
            </td>
          </tr>
          <tr>
            <td>Account-control evidence</td>
            <td><span className="badge">{String(control.status ?? "").replace(/_/g, " ")}</span></td>
          </tr>
          <tr>
            <td>Intent attribution</td>
            <td><span className="badge">{String(intent.status ?? "").replace(/_/g, " ")}</span></td>
          </tr>
        </tbody>
      </table>

      <div className="notice" style={{ marginTop: 10 }}>
        {card.limitation_statement}
      </div>

      <h3 style={{ marginTop: 16, fontSize: 13 }}>
        Evidence to verify before high-impact action
      </h3>
      <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
        {(card.verification_checklist ?? []).map((item: any) => (
          <li key={item.id} style={{ marginBottom: 4 }}>
            <input type="checkbox" disabled readOnly checked={false}
                   style={{ marginRight: 6 }} />
            {item.label}
            {item.status === "NOT_IN_THIS_DATASET" && (
              <span className="stat-sub"> — not held by this project</span>
            )}
            <div className="stat-sub" style={{ paddingLeft: 22 }}>{item.why}</div>
          </li>
        ))}
      </ul>
      <div className="stat-sub" style={{ marginTop: 8 }}>{card.checklist_note}</div>
    </div>
  );
}
