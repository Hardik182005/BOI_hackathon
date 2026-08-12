import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import CapacityOptimizer from "./pages/CapacityOptimizer";

// Real responses captured from a running backend, exactly as the other page
// tests do. The only edit is that the curve's `points` array is filtered down
// to the budgets the page needs to render - the rows themselves are untouched
// backend output, because the whole claim of this page is that the browser
// displays measured numbers rather than producing them.
import curve from "./__fixtures__/capacity_curve.json";
import plan from "./__fixtures__/capacity_plan.json";

const stub = () =>
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    let body: any = null;
    if (url === "/v1/capacity/curve") body = curve;
    else if (url === "/v1/capacity/plan") body = plan;
    if (body == null) return Promise.reject(new TypeError(`unmocked ${url}`));
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as any);
  }));

describe("Analyst Capacity Optimizer", () => {
  beforeEach(stub);

  it("offers both input modes and renders the measured curve", async () => {
    const { container } = render(<CapacityOptimizer />);
    await waitFor(() =>
      expect(screen.getByText(/Investigation capacity/)).toBeInTheDocument());
    expect(screen.getByText(/Accounts analysts can review per day/)).toBeInTheDocument();
    expect(screen.getByText(/Maximum acceptable false positives per 1,000/)).toBeInTheDocument();
    // Every headline row is the artifact's own row: the budget, its recall and
    // its interval have to appear together, or the page is showing something
    // it made up.
    const rows = Array.from(container.querySelectorAll("table.data tr"))
      .map((tr) => tr.textContent ?? "");
    for (const p of (curve as any).headline) {
      const want = `${p.budget} accounts`;
      const row = rows.find((t) => t.startsWith(want));
      expect(row, `no table row for budget ${p.budget}`).toBeTruthy();
      expect(row).toContain(`${(100 * p.recall).toFixed(1)}%`);
      expect(row).toContain(`${p.true_positives} of ${(curve as any).evaluation.n_positives}`);
    }
  });

  it("shows the answer and marks the threshold advisory, never applied", async () => {
    render(<CapacityOptimizer />);
    await waitFor(() =>
      expect(screen.getByText(/Investigation capacity/)).toBeInTheDocument());
    fireEvent.click(screen.getByText("Show what that buys"));

    const p: any = plan;
    await waitFor(() =>
      expect(screen.getByText(
        `Expected recall at ${p.answered_at_budget} reviews`)).toBeInTheDocument());
    // the displayed recall is the API's number to the digit
    const pct = `${(100 * p.expected.recall).toFixed(1)}%`;
    expect(screen.getAllByText(pct).length).toBeGreaterThan(0);
    expect(screen.getByText(/ADVISORY REQUIRES HUMAN APPROVAL/)).toBeInTheDocument();
    expect(screen.getByText(/applied: false/)).toBeInTheDocument();
    expect(screen.getByText(/Recommended operational threshold/)).toBeInTheDocument();
  });

  it("reports uncertainty rather than a single confident percentage", async () => {
    render(<CapacityOptimizer />);
    await waitFor(() =>
      expect(screen.getByText(/Investigation capacity/)).toBeInTheDocument());
    fireEvent.click(screen.getByText("Show what that buys"));
    await waitFor(() =>
      expect(screen.getByText(/How firm is this number\?/)).toBeInTheDocument());
    expect(screen.getByText(/whole book resampled/)).toBeInTheDocument();
    expect(screen.getByText(/Recall in each individual CV repeat/)).toBeInTheDocument();
    expect(screen.getByText(/What this number does not mean/)).toBeInTheDocument();
  });

  it("shows the frozen policy position without offering to change it", async () => {
    const { container } = render(<CapacityOptimizer />);
    await waitFor(() =>
      expect(screen.getByText(/Where the frozen policy already sits/)).toBeInTheDocument());
    for (const row of (curve as any).frozen_policy_position) {
      expect(screen.getByText(row.tier.replace(/_/g, " "))).toBeInTheDocument();
    }
    const buttons = Array.from(container.querySelectorAll("button"))
      .map((b) => (b.textContent ?? "").toLowerCase());
    expect(buttons.some((t) => t.includes("apply") || t.includes("save"))).toBe(false);
  });

  it("no forbidden vocabulary in the rendered page", async () => {
    const { container } = render(<CapacityOptimizer />);
    await waitFor(() =>
      expect(screen.getByText(/Investigation capacity/)).toBeInTheDocument());
    const text = (container.textContent ?? "")
      .replace("a behavioural risk score is not proof of criminal intent.", "")
      .toUpperCase();
    for (const bad of ["GUILTY", "CRIMINAL", "PERMANENTLY_SAFE",
                       "CERTIFIED_CLEAN", "AUTO_FREEZE"]) {
      expect(text.includes(bad)).toBe(false);
    }
  });
});
