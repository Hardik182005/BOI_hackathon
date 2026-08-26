import { render, screen, waitFor, within } from "@testing-library/react";
import { HashRouter } from "react-router-dom";
import { describe, it, expect, beforeEach, vi } from "vitest";
import App from "./App";
import lp from "./landing.css?raw";

// The landing page is the only screen in this project that is allowed to be
// persuasive, which is exactly why it needs a test: every figure on it has to
// be one the repository can produce an artifact for, and it must not quietly
// become a second place where the product claims something the console would
// refuse to claim.
const renderAt = (hash: string) => {
  window.location.hash = hash;
  return render(<HashRouter><App /></HashRouter>);
};

describe("public landing page", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("network down"))));
  });

  it("is the root route, and renders no analyst console around it", () => {
    const { container } = renderAt("#/");
    expect(container.querySelector(".lp")).toBeTruthy();
    // No sidebar, no nav, no `.main` - the landing page is full bleed.
    expect(container.querySelector(".sidebar")).toBeNull();
    expect(container.querySelector(".nav")).toBeNull();
    expect(container.querySelector(".main")).toBeNull();
  });

  it("shows the guardrail pipeline in place of a marketing diagram", () => {
    const { container } = renderAt("#/");
    const rail = container.querySelector(".lp-pipe-rail") as HTMLElement;
    expect(rail).toBeTruthy();
    // Six stages, and the ones that matter are the refusals.
    expect(rail.querySelectorAll(".lp-stage").length).toBe(6);
    for (const stage of [
      /Schema & leakage gate/, /Lens 1/, /Lens 2/, /Lens 3/,
      /Policy engine/, /Human analyst/,
    ]) {
      expect(within(rail).getByText(stage)).toBeInTheDocument();
    }
    expect(rail.textContent).toMatch(/422/);
    expect(rail.textContent).toMatch(/OOD_REVIEW/);
    expect(rail.textContent).toMatch(/no automatic freeze/);
  });

  it("never presents the walkthrough score as a measured result", () => {
    const { container } = renderAt("#/");
    const rail = container.querySelector(".lp-pipe-rail") as HTMLElement;
    // The one animated number on the page is a demonstration, and says so.
    expect(container.textContent).toMatch(/Illustrative walkthrough/i);
    expect(rail.textContent).toMatch(/A probability, not a verdict/);
    // Accuracy is the metric this project refuses to report, so the landing
    // page must not quietly become the one place it gets quoted.
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/accuracy/i);
  });

  it("quotes only the four sealed-test figures, in full", () => {
    const { container } = renderAt("#/");
    const stats = [...container.querySelectorAll(".lp-stat")];
    expect(stats.length).toBe(4);
    const text = stats.map((s) => s.textContent ?? "").join(" | ");
    // The locked-test PR-AUC, the lift over base rate and the calibration
    // error, each stated to the precision the artifact states it to.
    expect(text).toMatch(/0\.726/);
    expect(text).toMatch(/77\.7/);
    expect(text).toMatch(/0\.0015/);
    // And the one that is not a model metric at all.
    expect(text).toMatch(/frozen without a named analyst/);
  });

  it("leaves no in-page link pointing at a section that was removed", () => {
    const { container } = renderAt("#/");
    const anchors = [...container.querySelectorAll('a[href^="#"]')]
      .map((a) => a.getAttribute("href") ?? "")
      .filter((h) => h.length > 1 && !h.startsWith("#/"));
    expect(anchors.length).toBeGreaterThan(0);
    for (const href of anchors) {
      expect(container.querySelector(`section[id="${href.slice(1)}"]`),
             `dead anchor ${href}`).toBeTruthy();
    }
  });

  it("the console is still one click away and still intact", async () => {
    const { container } = renderAt("#/");
    const toConsole = [...container.querySelectorAll("a")]
      .filter((a) => (a.getAttribute("href") ?? "").endsWith("/overview"));
    expect(toConsole.length).toBeGreaterThan(0);

    const { container: c2 } = renderAt("#/overview");
    expect(c2.querySelectorAll(".nav a").length).toBe(10);
    await waitFor(() =>
      expect(c2.querySelector(".error-state, .loading")).toBeTruthy());
    expect(screen.queryByText(/Page not found/)).toBeNull();
  });

  it("never lets the anchor reset repaint a button label", () => {
    // `.lp a` is (0,1,1) and outranks a bare `.lp-btn-solid` (0,1,0), so an
    // unqualified reset won the cascade and painted the solid CTA label in
    // --ink on an --ink pill - a black blank button. Every anchor reset that
    // touches colour has to hold buttons out.
    const resets = [...lp.matchAll(/^\.lp a([^{,]*)\{([^}]*)\}/gm)];
    expect(resets.length).toBeGreaterThan(0);
    for (const [, qualifier, body] of resets) {
      if (!/color\s*:/.test(body)) continue;
      expect(qualifier, `.lp a${qualifier} repaints button labels`)
        .toMatch(/:not\([^)]*\.lp-btn/);
    }
    // ...and the solid button still declares a label colour of its own.
    expect(lp).toMatch(/\.lp-btn-solid\s*\{[^}]*color:\s*#fff/);
  });

  it("ships its own namespaced stylesheet and its own fonts", () => {
    // Every landing rule is scoped under `.lp`, so nothing here can reach the
    // console's white background and black text.
    const rules = lp
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/@font-face\s*\{[^}]*\}/g, "")
      .replace(/@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g, "")
      .replace(/@media[^{]*\{/g, "");
    const selectors = [...rules.matchAll(/([^{}]+)\{/g)]
      .map((m) => m[1].trim())
      .filter(Boolean);
    expect(selectors.length).toBeGreaterThan(100);
    for (const sel of selectors) {
      for (const part of sel.split(",")) {
        expect(part, `selector escapes the .lp namespace: ${sel}`)
          .toMatch(/\.lp\b|\.lp-/);
      }
    }
    // Fonts are served from this origin, so the page has no runtime network
    // dependency of any kind - the same guarantee the rest of the project makes.
    const faces = [...lp.matchAll(/@font-face\s*\{([^}]*)\}/g)].map((m) => m[1]);
    expect(faces.length).toBe(3);
    for (const face of faces) {
      const url = /src:\s*url\(["']?([^"')]+)/.exec(face);
      expect(url?.[1], `remote font: ${url?.[1]}`).toMatch(/^\/fonts\/.+\.woff2$/);
    }
    expect(lp, "landing.css fetches something at runtime")
      .not.toMatch(/url\(\s*["']?(https?:)?\/\//);
  });
});
