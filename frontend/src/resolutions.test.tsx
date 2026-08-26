import { render, waitFor } from "@testing-library/react";
import { HashRouter } from "react-router-dom";
import { vi, describe, it, expect, beforeEach } from "vitest";
import App from "./App";
import css from "./styles.css?raw";

import capacity from "./__fixtures__/capacity_curve.json";
import cases from "./__fixtures__/cases.json";
import metrics from "./__fixtures__/metrics.json";
import openapi from "./__fixtures__/openapi.json";
import pg from "./__fixtures__/pg.json";
import seals from "./__fixtures__/seals.json";

// The dashboard is claimed to work at 1280x720, 1366x768, 1440x900 and
// 1920x1080. This repo's frontend harness is vitest on jsdom, which builds a
// DOM but never lays it out - every box measures zero and no stylesheet is
// applied - so what a browser would call "overlap" cannot be measured here, and
// this file does not pretend otherwise. Three things can be established from
// this harness, and each test says which one it is:
//
//   1. every page renders its whole screen at each listed viewport, and the
//      navigation stays complete - a real check, because it would catch a page
//      that crashes or silently drops a section;
//   2. nothing in the app decides what to render from the window width, so the
//      four viewports differ only in CSS - which is what makes (3) the load
//      bearing check rather than a formality;
//   3. the stylesheet declares no breakpoint inside the supported range and no
//      fixed dimension wider than the narrowest content column, which is what a
//      clipped table or an off-screen button looks like before it is rendered.
//
// Pixel-level overlap at these resolutions needs a real engine; no browser
// driver is installed in this project and inventing one here would replace a
// measured claim with an assumed one.

const RESOLUTIONS: [number, number][] = [
  [1280, 720], [1366, 768], [1440, 900], [1920, 1080],
];

// From styles.css: .sidebar is 256px and .main pads 24px on each side. The
// content column at the narrowest supported width is what every fixed
// dimension in the app has to fit inside.
const SIDEBAR_PX = 256;
const MAIN_PADDING_PX = 48;
const NARROWEST_CONTENT_PX = Math.min(...RESOLUTIONS.map(([w]) => w))
  - SIDEBAR_PX - MAIN_PADDING_PX;

const NAV_LINKS = 10;

// Committed backend responses, the same ones the page tests use. A page whose
// endpoint has no fixture is still visited - it must show its error state, not
// a blank panel, which is one of the states section 33 asks to see.
const FIXTURES: Record<string, any> = {
  "/v1/metrics/summary": metrics,
  "/openapi.json": openapi,
  "/v1/validation/seals": seals,
  "/v1/capacity/curve": capacity,
  "/v1/graph/status": { status: "UNAVAILABLE" },
};

const stub = () =>
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    let body: any = null;
    if (FIXTURES[url]) body = FIXTURES[url];
    else if (url.startsWith("/v1/cases")) body = cases;
    else if (url.startsWith("/v1/proofgraph/")) body = pg;
    if (body == null) return Promise.reject(new TypeError(`unmocked ${url}`));
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as any);
  }));

// `heading` is the page's own title. It is null where no response for that
// page's endpoint has been captured from a running backend - drift status is
// the only one - and hand-writing that payload to reach a title would make the
// assertion about the fixture rather than about the page.
const ROUTES: { hash: string; heading: RegExp | null }[] = [
  { hash: "#/overview", heading: /Executive Overview/ },
  { hash: "#/queue", heading: /Alert Queue/ },
  { hash: "#/performance", heading: /Model Performance/ },
  { hash: "#/features", heading: /Feature Intelligence/ },
  { hash: "#/model-card", heading: /Model Card/ },
  { hash: "#/validation", heading: /Validation Lab/ },
  { hash: "#/value", heading: /Business Value/ },
  { hash: "#/capacity", heading: /Analyst Capacity Optimizer/ },
  { hash: "#/graph", heading: /Graph Lab/ },
  { hash: "#/proof/CASE-45C8DE5165", heading: /ProofGraph/ },
  { hash: "#/drift", heading: null },
];

function setViewport(width: number, height: number) {
  for (const [k, v] of [["innerWidth", width], ["innerHeight", height]] as const) {
    Object.defineProperty(window, k, { value: v, writable: true, configurable: true });
  }
  window.dispatchEvent(new Event("resize"));
}

// Shipped source only, keyed by path. The test files are excluded for the same
// reason the QA harness excludes them: a test that asserts a pattern is absent
// has to contain the pattern to say so.
const SOURCES: [string, string][] = Object.entries(
  import.meta.glob("./**/*.tsx", { query: "?raw", import: "default", eager: true }) as Record<string, string>,
).filter(([p]) => !p.includes(".test.") && !p.split("/").pop()!.startsWith("__"));

describe("desktop resolutions (section 33)", () => {
  beforeEach(stub);

  for (const [w, h] of RESOLUTIONS) {
    it(`every page renders a complete screen at ${w}x${h}`, async () => {
      setViewport(w, h);
      for (const route of ROUTES) {
        window.location.hash = route.hash;
        const { container, unmount } = render(<HashRouter><App /></HashRouter>);
        const main = container.querySelector(".main") as HTMLElement;

        if (route.heading) {
          await waitFor(() =>
            expect(main.querySelector(".page-title")?.textContent ?? "",
                   `${route.hash} at ${w}x${h}`).toMatch(route.heading!));
        } else {
          await waitFor(() => expect(main.querySelector(".error-state"),
                                     `${route.hash} at ${w}x${h}`).toBeTruthy());
        }
        // A screen that renders a heading and nothing else is still a blank
        // screen to the analyst looking at it.
        expect((main.textContent ?? "").length,
               `${route.hash} at ${w}x${h}`).toBeGreaterThan(120);
        expect(main.textContent).not.toContain("Page not found");
        expect(container.querySelectorAll(".nav a").length).toBe(NAV_LINKS);
        unmount();
      }
    });
  }

  it("no page chooses what to render from the window size", () => {
    for (const [path, src] of SOURCES) {
      expect(src, path).not.toMatch(/window\.(innerWidth|innerHeight|matchMedia)/);
      expect(src, path).not.toMatch(/useMediaQuery/);
    }
  });

  it("no stylesheet breakpoint falls inside the supported range", () => {
    const breakpoints = [...css.matchAll(/@media[^{]*max-width\s*:\s*(\d+(?:\.\d+)?)px/g)]
      .map((m) => Number(m[1]));
    expect(breakpoints.length).toBeGreaterThan(0);
    // A breakpoint at or above the narrowest supported width would mean one of
    // the four resolutions gets the stacked small-screen layout - the sidebar
    // on top of the content - which is not the layout being claimed.
    for (const bp of breakpoints) {
      expect(bp, `@media max-width ${bp}px`).toBeLessThan(NARROWEST_CONTENT_PX + SIDEBAR_PX);
    }
  });

  it("nothing declares a width the narrowest content column cannot hold", () => {
    const declared: [string, number][] = [];
    for (const m of css.matchAll(/(^|[\s;{])(min-width|width)\s*:\s*(\d+(?:\.\d+)?)px/g)) {
      declared.push([`styles.css ${m[2]}`, Number(m[3])]);
    }
    // JSX inline styles: the camelCase spelling only ever appears inside a
    // style object, so SVG geometry attributes (which are viewBox units and
    // scale with the canvas) are not swept up by this.
    for (const [path, src] of SOURCES) {
      for (const m of src.matchAll(/style=\{\{[^}]*?\b(minWidth|width)\s*:\s*(\d+(?:\.\d+)?)\b/g)) {
        declared.push([`${path} ${m[1]}`, Number(m[2])]);
      }
    }
    expect(declared.length).toBeGreaterThan(0);
    for (const [where, px] of declared) {
      expect(px, `${where}: ${px}px`).toBeLessThanOrEqual(NARROWEST_CONTENT_PX);
    }
  });

  it("the wide panels scroll inside their own box rather than the page", async () => {
    // The evidence graph is the one canvas drawn to a fixed coordinate system,
    // so it is the one that would push the page sideways if it were not held in
    // a scrolling container. Asserted from the rendered DOM at the narrowest
    // supported width, not from the source.
    setViewport(1280, 720);
    window.location.hash = "#/proof/CASE-45C8DE5165";
    const { container } = render(<HashRouter><App /></HashRouter>);
    const svg = await waitFor(() => {
      const el = container.querySelector("svg[aria-label='evidence graph']");
      expect(el).toBeTruthy();
      return el as SVGSVGElement;
    });
    expect(svg.getAttribute("viewBox")).toBeTruthy();
    expect(svg.getAttribute("width")).toBe("100%");
    expect(svg.closest(".graph-canvas")).toBeTruthy();
    expect(css).toMatch(/\.graph-canvas\s*\{[^}]*overflow-x:\s*auto/);
    // Every table wide enough to need it is wrapped the same way.
    expect(css).toMatch(/table\.data\s*\{[^}]*width:\s*100%/);
  });
});
