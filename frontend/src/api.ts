// Thin API client. Every number shown in the UI comes from these endpoints,
// which serve pipeline artifacts - the frontend never invents data.
export type Json = Record<string, any>;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function unwrap(res: Response): Promise<Json> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, String(detail));
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
};

export function fmtPct(x: number | null | undefined, digits = 1): string {
  return x == null ? "–" : `${(100 * x).toFixed(digits)}%`;
}
export function fmtNum(x: number | null | undefined, digits = 3): string {
  return x == null ? "–" : x.toFixed(digits);
}
