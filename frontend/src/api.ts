// Thin API client. Every number shown in the UI comes from these endpoints,
// which serve pipeline artifacts - the frontend never invents data.
export type Json = Record<string, any>;

export class ApiError extends Error {
  // `detail` is the raw FastAPI error body (string, for most routes, or a
  // structured object for the ProofGraph retired/inadmissible 409s) - kept
  // alongside the human `message` so a caller that needs the structure
  // (error code, stored vs current model_version, ...) doesn't have to
  // re-parse text this class already parsed once.
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
  }
}

// FastAPI's `detail` is usually a plain string, but a few routes (the
// retired-evidence 409) send an object with a `message` field for humans and
// the rest for callers that want the structure. `String(detail)` on an
// object renders the literal text "[object Object]"; this picks the human
// sentence out of it instead, and falls back to the raw JSON only for an
// object shape nobody has written a `message` for yet.
function describeDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && typeof (detail as any).message === "string") {
    return (detail as any).message;
  }
  try {
    return detail == null ? fallback : JSON.stringify(detail);
  } catch {
    return fallback;
  }
}

async function unwrap(res: Response): Promise<Json> {
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, describeDetail(detail, res.statusText), detail);
  }
  try {
    return await res.json();
  } catch {
    // A 200 that is not JSON means something other than the API answered -
    // usually a dev-server or proxy fallback. Say so rather than crashing.
    throw new ApiError(res.status, "the response was not JSON - check that the API proxy is routing this path to the backend");
  }
}

async function request(path: string, init?: RequestInit): Promise<Json> {
  let res: Response;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(0, "API unreachable - is the backend running on :8000?");
  }
  return unwrap(res);
}

// The endpoint's own bytes, unparsed. A file rebuilt from React state is only
// as trustworthy as the component that held it; this one is the response the
// page was drawn from, which is what makes a downloaded ProofGraph evidence.
async function requestText(path: string): Promise<string> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" } });
  } catch {
    throw new ApiError(0, "API unreachable - is the backend running on :8000?");
  }
  const body = await res.text();
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      detail = JSON.parse(body).detail ?? detail;
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, describeDetail(detail, res.statusText), detail);
  }
  return body;
}

// Multipart upload. The Content-Type header is deliberately not set: the
// browser has to add the multipart boundary itself, and overriding it makes
// FastAPI reject the body before it ever reaches a route.
async function upload(path: string, form: FormData): Promise<Json> {
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", body: form });
  } catch {
    throw new ApiError(0, "API unreachable - is the backend running on :8000?");
  }
  return unwrap(res);
}

export const api = {
  health: () => request("/health/ready"),
  model: () => request("/v1/model"),
  metrics: () => request("/v1/metrics/summary"),
  drift: () => request("/v1/drift/status"),
  cases: (params = "") => request(`/v1/cases${params}`),
  caseDetail: (id: string) => request(`/v1/cases/${id}`),
  decision: (id: string, body: Json) =>
    request(`/v1/cases/${id}/decision`, { method: "POST", body: JSON.stringify(body) }),
  feedback: (id: string, body: Json) =>
    request(`/v1/cases/${id}/feedback`, { method: "POST", body: JSON.stringify(body) }),
  generateReport: (id: string, useLlm: boolean) =>
    request(`/v1/reports/${id}/generate?use_llm=${useLlm}`, { method: "POST" }),
  score: (body: Json) =>
    request("/v1/score", { method: "POST", body: JSON.stringify(body) }),

  // Sealed hidden-validation workflow. `run` never returns a metric; `reveal`
  // is a separate call that a person has to make deliberately.
  validationRun: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return upload("/v1/validation/run", form);
  },
  validationReveal: (sealId: string, file: File, labelColumn?: string) => {
    const form = new FormData();
    form.append("file", file);
    const q = labelColumn ? `?label_column=${encodeURIComponent(labelColumn)}` : "";
    return upload(`/v1/validation/${encodeURIComponent(sealId)}/reveal${q}`, form);
  },
  seals: () => request("/v1/validation/seals"),
  seal: (sealId: string) => request(`/v1/validation/seals/${encodeURIComponent(sealId)}`),

  // Analyst Capacity Optimizer. The curve is a precomputed artifact and the
  // plan is a lookup inside it, so the browser never computes a metric.
  capacityCurve: () => request("/v1/capacity/curve"),
  capacityPlan: (body: Json) =>
    request("/v1/capacity/plan", { method: "POST", body: JSON.stringify(body) }),

  // Dual-evidence ProofGraph.
  proofgraph: (caseId: string, twin = true) =>
    request(`/v1/proofgraph/${encodeURIComponent(caseId)}?twin=${twin}`),
  proofgraphJson: (caseId: string, twin = true) =>
    requestText(`/v1/proofgraph/${encodeURIComponent(caseId)}?twin=${twin}`),

  // Used by Graph Lab to ask the running backend which routes actually exist,
  // rather than assuming a counterparty-graph endpoint is deployed.
  openapi: () => request("/openapi.json"),

  // Optional transaction-graph adapter. Every one of these answers
  // status: "UNAVAILABLE" until a real edge file is uploaded, which is the
  // truthful state for the competition dataset.
  graphStatus: () => request("/v1/graph/status"),
  graphUpload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return upload("/v1/graph/edges", form);
  },
  graphDiscard: () => request("/v1/graph/edges", { method: "DELETE" }),
  graphAccount: (account: string) =>
    request(`/v1/graph/account/${encodeURIComponent(account)}`),

  // Trinetra Mule-Farm Cohort Radar. Retrieval over a frozen development-only
  // reference - these routes read the classifier's output and never change it.
  cohortManifest: () => request("/v1/cohort/manifest"),
  cohortForCase: (caseId: string, k = 10) =>
    request(`/v1/cases/${encodeURIComponent(caseId)}/cohort?k=${k}`),
  cohortSearch: (body: Json) =>
    request("/v1/cohort/search", { method: "POST", body: JSON.stringify(body) }),
};

/** The locked-test headline, attributed to the model that actually serves.
 *
 * `locked_test` in the metrics summary is the stored evaluation artifact, and
 * that artifact belongs to `catboost_tuned_top60` - a model the Feature
 * Availability Firewall retired. The backend annotates the block
 * (`describes_deployed_model`, `deployed_scorer_result`, `retired_warning`)
 * rather than overwriting it, because its tier and calibration sections have no
 * post-firewall equivalent yet. So the ranking headline has to be resolved here,
 * once, instead of in every page: a dashboard that prints a retired model's
 * PR-AUC as its primary metric is stating a number no judge could reproduce
 * against the running scorer.
 *
 * `deployed_scorer_result.pr_auc` is a bare float; the retired block nests it
 * under `.point` with a bootstrap interval. Both shapes are normalised here.
 */
export type LockedHeadline = {
  prAuc?: number;
  prAucCi?: [number | undefined, number | undefined];
  rocAuc?: number;
  model?: string;
  n?: number;
  nPositives?: number;
  /** false when the served artifact belongs to a retired model */
  fromDeployedModel: boolean;
  retiredModelWarning?: string;
};

export function lockedHeadline(lt: Json | null | undefined): LockedHeadline {
  const num = (x: any): number | undefined =>
    typeof x === "number" ? x : (typeof x?.point === "number" ? x.point : undefined);
  if (!lt) return { fromDeployedModel: true };

  const deployed = lt.deployed_scorer_result;
  if (lt.describes_deployed_model === false && deployed) {
    return {
      prAuc: num(deployed.pr_auc),
      // The sealed single-touch evaluation stores no bootstrap interval, and
      // inventing one from the retired run would be worse than showing none.
      prAucCi: undefined,
      rocAuc: num(deployed.roc_auc),
      model: deployed.model,
      n: deployed.n,
      nPositives: deployed.n_positives,
      fromDeployedModel: false,
      retiredModelWarning: lt.retired_warning,
    };
  }
  return {
    prAuc: num(lt.pr_auc),
    prAucCi: [lt.pr_auc?.ci_low, lt.pr_auc?.ci_high],
    rocAuc: num(lt.roc_auc),
    n: lt.n,
    nPositives: lt.n_positives,
    fromDeployedModel: true,
  };
}

export function fmtCi(ci: [number | undefined, number | undefined] | undefined,
                      digits = 3): string {
  if (!ci || ci[0] == null || ci[1] == null) return "no bootstrap interval stored";
  return `95% CI ${fmtNum(ci[0], digits)}–${fmtNum(ci[1], digits)}`;
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  return x == null ? "–" : `${(100 * x).toFixed(digits)}%`;
}
export function fmtNum(x: number | null | undefined, digits = 3): string {
  return x == null ? "–" : x.toFixed(digits);
}
