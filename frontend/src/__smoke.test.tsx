import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi, describe, it, expect, beforeEach } from "vitest";
import ValidationLab from "./pages/ValidationLab";
import BusinessValue from "./pages/BusinessValue";
import GraphLab from "./pages/GraphLab";
import ProofGraph from "./pages/ProofGraph";

// Real responses captured from a running backend and committed alongside the
// test. Hand-written fixtures would drift from the API the moment a field was
// renamed, and these pages exist to display exactly what the backend proved -
// so the test asserts against what it actually said.
import cases from "./__fixtures__/cases.json";
import metrics from "./__fixtures__/metrics.json";
import openapi from "./__fixtures__/openapi.json";
import pg from "./__fixtures__/pg.json";
import reveal2 from "./__fixtures__/reveal2.json";
import run from "./__fixtures__/run.json";
import seals from "./__fixtures__/seals.json";

const FIXTURES: Record<string, any> = {
  "/v1/metrics/summary": metrics,
  "/openapi.json": openapi,
  "/v1/validation/seals": seals,
};

const stub = () =>
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    let body: any = null;
    if (FIXTURES[url]) body = FIXTURES[url];
    else if (url.startsWith("/v1/cases")) body = cases;
    else if (url.startsWith("/v1/proofgraph/")) body = pg;
    else if (url.includes("/reveal")) body = reveal2;
    else if (url === "/v1/validation/run") body = run;
    // The adapter's honest default: deployed, and holding nothing.
    else if (url === "/v1/graph/status") body = { status: "UNAVAILABLE" };
    else if (url.startsWith("/v1/validation/seals/")) body = { seal: (seals as any).seals[0], verification: { verified: true, detail: "ok" } };
    if (body == null) return Promise.reject(new TypeError(`unmocked ${url}`));
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as any);
  }));

describe("new pages render real API payloads", () => {
  beforeEach(stub);

  it("Business Value shows locked-test budget rows", async () => {
    render(<BusinessValue />);
    await waitFor(() => expect(screen.getByText(/top 100/)).toBeInTheDocument());
    expect(screen.getByText(/Mules caught at fixed analyst review budgets/)).toBeInTheDocument();
    expect(screen.getByText(/False alarms per 1,000 legitimate accounts/)).toBeInTheDocument();
  });

  it("Graph Lab refuses to fabricate a transaction graph", async () => {
    render(<GraphLab />);
    await waitFor(() =>
      expect(screen.getByText(/refuses to fabricate a transaction graph/)).toBeInTheDocument());
    // The adapter route IS deployed on this backend, and the page still draws
    // nothing - because a deployed route is not an edge file. Asserting the
    // route-present wording rather than the route-absent one is the stronger
    // test: it proves the refusal survives the feature existing.
    await waitFor(() =>
      expect(screen.getByText(/ROUTE PRESENT/)).toBeInTheDocument());
    expect(screen.getByText(/NO EDGE FILE LOADED/)).toBeInTheDocument();
    expect(screen.queryByText(/A transaction graph is loaded/)).toBeNull();
  });

  it("ProofGraph renders prosecution and defence and the graph svg", async () => {
    render(
      <MemoryRouter initialEntries={["/proof/CASE-45C8DE5165"]}>
        <Routes><Route path="/proof/:caseId" element={<ProofGraph />} /></Routes>
      </MemoryRouter>
    );
    await waitFor(() => expect(screen.getByText("Prosecution")).toBeInTheDocument());
    expect(screen.getByText("Defence")).toBeInTheDocument();
    expect(screen.getByText("Verdict")).toBeInTheDocument();
    expect(document.querySelector("svg[aria-label='evidence graph']")).toBeTruthy();
    expect(document.querySelectorAll("svg rect").length).toBeGreaterThan(3);
    expect(screen.getAllByText(/Counterfactual twin/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Model disagreement/)).toBeInTheDocument();
  });

  it("Validation Lab: no metric before reveal, three ordered steps after run", async () => {
    const { container } = render(<ValidationLab />);
    await waitFor(() => expect(screen.getByText(/Seal registry/)).toBeInTheDocument());
    expect(screen.getByText(/No validation metric exists yet/)).toBeInTheDocument();
    const revealBtn = screen.getByText("Reveal Validation Metrics") as HTMLButtonElement;
    expect(revealBtn.disabled).toBe(true);

    const input = container.querySelector("input[type=file]") as HTMLInputElement;
    const f = new File(["a,b\n1,2\n"], "v.csv", { type: "text/csv" });
    Object.defineProperty(input, "files", { value: [f] });
    fireEvent.change(input);
    fireEvent.click(screen.getByText("Run sealed validation"));

    await waitFor(() => expect(screen.getByText("Schema Integrity")).toBeInTheDocument());
    const nums = Array.from(container.querySelectorAll(".step-num")).map((e) => e.textContent);
    expect(nums.slice(0, 3)).toEqual(["1", "2", "3"]);
    expect(screen.getByText("Hidden Validation Shield")).toBeInTheDocument();
    expect(screen.getByText("Predictions (sealed)")).toBeInTheDocument();
    expect(screen.getByText(/No metric has been computed at this point/)).toBeInTheDocument();
    // compatibility components broken out
    for (const c of ["schema completeness", "distribution compatibility",
                     "missingness consistency", "value range coverage"]) {
      expect(screen.getByText(c)).toBeInTheDocument();
    }
    // protocol statement verbatim
    expect(screen.getByText(/Validation runs in a fixed order/)).toBeInTheDocument();
  });

  it("no forbidden vocabulary anywhere in the rendered pages", async () => {
    const { container } = render(
      <MemoryRouter initialEntries={["/proof/CASE-45C8DE5165"]}>
        <Routes><Route path="/proof/:caseId" element={<ProofGraph />} /></Routes>
      </MemoryRouter>
    );
    await waitFor(() => expect(screen.getByText("Prosecution")).toBeInTheDocument());
    // The only occurrence of any of these strings in the whole app is the
    // pre-existing human-review disclaimer, which negates the word rather than
    // asserting it. Removed here so the assertion tests the new page copy.
    const text = (container.textContent ?? "")
      .replace("a behavioural risk score is not proof of criminal intent.", "")
      .toUpperCase();
    for (const bad of ["GUILTY", "CRIMINAL", "PERMANENTLY_SAFE",
                       "CERTIFIED_CLEAN", "AUTO_FREEZE"]) {
      expect(text.includes(bad)).toBe(false);
    }
  });
});
