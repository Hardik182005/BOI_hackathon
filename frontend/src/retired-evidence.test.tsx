import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi, describe, it, expect } from "vitest";
import CaseDetail from "./pages/CaseDetail";
import ProofGraph from "./pages/ProofGraph";
import { api, ApiError } from "./api";

// Feature Availability Firewall retirement: four stored cases were scored by
// model_version "1.0.0" (a CatBoost bundle frozen before the firewall
// existed), and their payloads quarantine columns F3898/F3914 in
// top_reasons. The backend now withholds top_reasons/counterfactual_twin for
// those cases (GET /v1/cases/{id} carries `evidence_status`) and refuses the
// ProofGraph outright with a structured 409. These tests pin the UI half:
// the retired banner appears, the drivers table it replaces never renders,
// and a dict-valued `detail` reads as a sentence rather than "[object Object]".

const RETIRED_409_DETAIL = {
  error: "RETIRED_EVIDENCE",
  case_id: "CASE-18A744455E",
  status: "RETIRED",
  stored_model_version: "1.0.0",
  current_model_version: "2.0.0",
  message: "This case was scored by model_version '1.0.0', which is no longer "
    + "the production model (now '2.0.0'); see "
    + "/v1/proofgraph/CASE-18A744455E/provenance for what it was built from.",
};

const RETIRED_CASE = {
  case: {
    case_id: "CASE-18A744455E", account_reference: "DEMO-HIGH_RISK_MULE",
    status: "OPEN", risk_tier: "CRITICAL_REVIEW", calibrated_risk: 0.999,
  },
  score: {
    model_agreement: 0.9, conformal_status: "HIGH_RISK_SET", ood_status: "IN_DISTRIBUTION",
    anomaly_percentile: 40, raw_scores: { xgboost: 0.9 }, verifier_confirms_risk: true,
    reasons: ["retained for audit only"],
    // top_reasons and counterfactual_twin are dropped entirely, not emptied.
    evidence_withheld: {
      reason: "RETIRED_EVIDENCE",
      explanation: RETIRED_409_DETAIL.message,
      withheld_keys: ["top_reasons", "counterfactual_twin"],
      quarantined_features_used: ["F3898", "F3914"],
      audit_record_available_at: "/v1/proofgraph/CASE-18A744455E/provenance",
    },
  },
  actions: [], feedback: [], control_attribution: null,
  evidence_status: {
    admissible_as_current_evidence: false,
    reason: "RETIRED_EVIDENCE",
    explanation: RETIRED_409_DETAIL.message,
    provenance: { status: "RETIRED", stored_model_version: "1.0.0", current_model_version: "2.0.0" },
    quarantined_features_used: ["F3898", "F3914"],
  },
};

const CURRENT_CASE = {
  case: {
    case_id: "CASE-CURRENT", account_reference: "DEMO-CURRENT",
    status: "OPEN", risk_tier: "MONITOR", calibrated_risk: 0.1,
  },
  score: {
    model_agreement: 0.95, conformal_status: "OK", ood_status: "IN_DISTRIBUTION",
    anomaly_percentile: 50, raw_scores: { xgboost: 0.1 }, verifier_confirms_risk: false,
    reasons: [],
    top_reasons: [{
      feature: "F1", verified_semantic_name: "Foo", value: 1,
      legitimate_percentile: 50, direction: "INCREASES_RISK", shap_contribution: 0.1,
    }],
  },
  actions: [], feedback: [], control_attribution: null,
  evidence_status: {
    admissible_as_current_evidence: true, reason: null, explanation: null,
    provenance: { status: "CURRENT", stored_model_version: "2.0.0", current_model_version: "2.0.0" },
    quarantined_features_used: [],
  },
};

describe("CaseDetail — retired evidence banner", () => {
  it("shows the RETIRED/STALE banner and hides the drivers table for an inadmissible case", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) =>
      url === "/v1/cases/CASE-18A744455E"
        ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(RETIRED_CASE) } as any)
        : Promise.reject(new TypeError(`unmocked ${url}`))
    ));
    render(
      <MemoryRouter initialEntries={["/cases/CASE-18A744455E"]}>
        <Routes><Route path="/cases/:caseId" element={<CaseDetail />} /></Routes>
      </MemoryRouter>
    );
    await waitFor(() => expect(screen.getByText(/RETIRED \/ STALE EVIDENCE/)).toBeInTheDocument());
    // the backend's own explanation, not a UI paraphrase of it
    expect(screen.getByText(/no longer the production model/)).toBeInTheDocument();

    // the drivers table (the one with a SHAP column) must not render at all -
    // it is the only <thead> on this page that would ever say "SHAP"
    const theadText = Array.from(document.querySelectorAll("thead")).map((t) => t.textContent).join(" ");
    expect(theadText).not.toMatch(/SHAP/);

    // naming which columns were quarantined is disclosure, not reconstruction:
    // it's fine for the banner to say F3898/F3914 were involved, so long as no
    // value/percentile/SHAP number for them is ever rendered as a live driver
    const panel = screen.getByText(/RETIRED \/ STALE EVIDENCE/).closest(".notice") as HTMLElement;
    expect(within(panel).getByText("F3898")).toBeInTheDocument();
    expect(within(panel).getByText("F3914")).toBeInTheDocument();

    // the rest of the case file stays visible and reachable
    expect(screen.getByText("Analyst actions")).toBeInTheDocument();
    expect(screen.getByText(/Analyst feedback/)).toBeInTheDocument();
    expect(screen.getByText("CRITICAL REVIEW")).toBeInTheDocument();
  });

  it("renders the drivers table normally for a current, admissible case", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) =>
      url === "/v1/cases/CASE-CURRENT"
        ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(CURRENT_CASE) } as any)
        : Promise.reject(new TypeError(`unmocked ${url}`))
    ));
    render(
      <MemoryRouter initialEntries={["/cases/CASE-CURRENT"]}>
        <Routes><Route path="/cases/:caseId" element={<CaseDetail />} /></Routes>
      </MemoryRouter>
    );
    await waitFor(() => expect(screen.getByText("Foo")).toBeInTheDocument());
    expect(screen.queryByText(/RETIRED \/ STALE EVIDENCE/)).toBeNull();
  });
});

describe("ProofGraph — retired evidence 409", () => {
  it("renders a labelled retired panel instead of the generic ErrorState", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) =>
      url.startsWith("/v1/proofgraph/CASE-18A744455E")
        ? Promise.resolve({
            ok: false, status: 409, statusText: "Conflict",
            json: () => Promise.resolve({ detail: RETIRED_409_DETAIL }),
          } as any)
        : Promise.reject(new TypeError(`unmocked ${url}`))
    ));
    render(
      <MemoryRouter initialEntries={["/proof/CASE-18A744455E"]}>
        <Routes><Route path="/proof/:caseId" element={<ProofGraph />} /></Routes>
      </MemoryRouter>
    );
    await waitFor(() => expect(screen.getByText(/RETIRED \/ STALE EVIDENCE/)).toBeInTheDocument());
    expect(screen.getByText(/no longer the production model/)).toBeInTheDocument();
    expect(screen.getAllByText(/provenance/).length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toContain("[object Object]");
    expect(document.querySelector(".error-state")).toBeNull();
  });

  it("still uses the generic ErrorState for a 409 that isn't retired evidence", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) =>
      url.startsWith("/v1/proofgraph/CASE-OTHER")
        ? Promise.resolve({
            ok: false, status: 409, statusText: "Conflict",
            json: () => Promise.resolve({ detail: "some unrelated conflict" }),
          } as any)
        : Promise.reject(new TypeError(`unmocked ${url}`))
    ));
    render(
      <MemoryRouter initialEntries={["/proof/CASE-OTHER"]}>
        <Routes><Route path="/proof/:caseId" element={<ProofGraph />} /></Routes>
      </MemoryRouter>
    );
    await waitFor(() => expect(document.querySelector(".error-state")).toBeTruthy());
    expect(screen.getByText("some unrelated conflict")).toBeInTheDocument();
    expect(screen.queryByText(/RETIRED \/ STALE EVIDENCE/)).toBeNull();
  });
});

describe("api.ts — structured error detail (never \"[object Object]\")", () => {
  it("resolves a dict-valued 409 detail to its human message", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({
      ok: false, status: 409, statusText: "Conflict",
      json: () => Promise.resolve({ detail: RETIRED_409_DETAIL }),
    } as any)));
    await expect(api.proofgraph("CASE-X")).rejects.toMatchObject({ message: RETIRED_409_DETAIL.message });
  });

  it("attaches the parsed object to the thrown error for callers that want the structure", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({
      ok: false, status: 409, statusText: "Conflict",
      json: () => Promise.resolve({ detail: RETIRED_409_DETAIL }),
    } as any)));
    expect.assertions(3);
    try {
      await api.proofgraph("CASE-X");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).detail).toEqual(RETIRED_409_DETAIL);
      expect((e as ApiError).message).not.toContain("[object Object]");
    }
  });

  it("keeps rendering a plain string detail unchanged", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({
      ok: false, status: 404, statusText: "Not Found",
      json: () => Promise.resolve({ detail: "case not found" }),
    } as any)));
    await expect(api.proofgraph("CASE-X")).rejects.toMatchObject({ message: "case not found" });
  });

  it("falls back to statusText for a non-JSON error body", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({
      ok: false, status: 500, statusText: "Internal Server Error",
      json: () => Promise.reject(new SyntaxError("not json")),
    } as any)));
    await expect(api.proofgraph("CASE-X")).rejects.toMatchObject({ message: "Internal Server Error" });
  });
});
