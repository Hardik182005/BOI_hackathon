// Thin API client. Every number shown in the UI comes from these endpoints,
// which serve pipeline artifacts - the frontend never invents data.
export type Json = Record<string, any>;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
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
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, String(detail));
  }
  return res.json();
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
};

export function fmtPct(x: number | null | undefined, digits = 1): string {
  return x == null ? "–" : `${(100 * x).toFixed(digits)}%`;
}
export function fmtNum(x: number | null | undefined, digits = 3): string {
  return x == null ? "–" : x.toFixed(digits);
}
