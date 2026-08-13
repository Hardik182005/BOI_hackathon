import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi, describe, it, expect, beforeEach } from "vitest";
import Overview from "./pages/Overview";
import BusinessValue from "./pages/BusinessValue";
import ModelPerformance from "./pages/ModelPerformance";
import ModelCard from "./pages/ModelCard";
import { lockedHeadline } from "./api";

// The dashboard's headline PR-AUC used to be read straight out of
// `locked_test.pr_auc.point`. That field belongs to `catboost_tuned_top60`, a
// model the leakage firewall retired: it was tuned on a feature pool that
// contained quarantined columns, and it is not what the API scores with. So the
// number on the front page - 0.8242 - was one no judge could reproduce against
// the running system, presented as the primary metric.
//
// These tests pin the fix at the only place it matters: what the page renders.
import cases from "./__fixtures__/cases.json";
import metrics from "./__fixtures__/metrics.json";

const RETIRED = "0.824";   // catboost_tuned_top60, pre-firewall
const DEPLOYED = "0.726";  // xgboost_top_120, the scorer actually served

const stub = () =>
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    let body: any = null;
    if (url === "/v1/metrics/summary") body = metrics;
    else if (url === "/v1/model") body = { winner: "xgboost_top_120", model_version: "2.0.0", n_features: 120 };
    else if (url.startsWith("/v1/cases")) body = cases;
    else if (url === "/health/ready") body = { status: "ready" };
    else if (url === "/v1/drift/status") body = { status: "STABLE" };
    if (body == null) return Promise.reject(new TypeError(`unmocked ${url}`));
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as any);
  }));

const wrap = (el: JSX.Element) => render(<MemoryRouter>{el}</MemoryRouter>);

describe("locked-test attribution", () => {
  beforeEach(stub);

  it("the fixture really is the retired artifact (or these tests prove nothing)", () => {
    const lt = (metrics as any).locked_test;
    expect(lt.pr_auc.point.toFixed(3)).toBe(RETIRED);
    expect(lt.describes_deployed_model).toBe(false);
    expect(lt.deployed_scorer_result.model).toBe("xgboost_top_120");
  });

  it("resolves the headline to the deployed scorer, and drops the retired interval", () => {
    const head = lockedHeadline((metrics as any).locked_test);
    expect(head.fromDeployedModel).toBe(false);
    expect(head.prAuc?.toFixed(3)).toBe(DEPLOYED);
    // The retired run's bootstrap interval does not belong to this point
    // estimate. Showing none is honest; reusing that one is not.
    expect(head.prAucCi).toBeUndefined();
  });

  it("passes an unannotated block through unchanged", () => {
    const head = lockedHeadline({ pr_auc: { point: 0.5, ci_low: 0.4, ci_high: 0.6 }, roc_auc: { point: 0.9 } });
    expect(head.fromDeployedModel).toBe(true);
    expect(head.prAuc).toBe(0.5);
    expect(head.prAucCi).toEqual([0.4, 0.6]);
  });

  it("survives a missing locked-test block without inventing a number", () => {
    const head = lockedHeadline(undefined);
    expect(head.prAuc).toBeUndefined();
    expect(head.fromDeployedModel).toBe(true);
  });

  for (const [name, el] of [
    ["Executive Overview", <Overview />],
    ["Business Value", <BusinessValue />],
    ["Model Performance", <ModelPerformance />],
    ["Model Card", <ModelCard />],
  ] as const) {
    it(`${name} shows the deployed PR-AUC and never the retired one`, async () => {
      const { container } = wrap(el);
      await waitFor(() =>
        expect(container.querySelector(".stat, table.data")).toBeTruthy());
      const text = container.textContent ?? "";
      expect(text).toContain(DEPLOYED);
      expect(text).not.toContain(RETIRED);
    });

    it(`${name} says out loud that the stored artifact is retired`, async () => {
      wrap(el);
      await waitFor(() =>
        expect(screen.getByText(/Read this before the numbers/)).toBeInTheDocument());
      // "retired" may appear more than once on a page; one is enough.
      expect(screen.getAllByText(/retired/i).length).toBeGreaterThan(0);
    });
  }
});
