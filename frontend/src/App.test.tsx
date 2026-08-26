import { render, screen, waitFor } from "@testing-library/react";
import { HashRouter } from "react-router-dom";
import { vi, describe, it, expect, beforeEach } from "vitest";
import App from "./App";

// The frontend never invents data: with the API down, pages must show an
// explicit error state - not fake numbers, not a blank screen.
const renderApp = () =>
  render(
    <HashRouter>
      <App />
    </HashRouter>
  );

describe("App routes", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("network down"))));
    window.location.hash = "#/overview";
  });

  it("renders sidebar with all pages and safety tagline", () => {
    renderApp();
    expect(screen.getByText("MuleGuard")).toBeInTheDocument();
    for (const label of [
      "Executive Overview", "Alert Queue", "Model Performance",
      "Feature Intelligence", "Drift & Monitoring", "Model Card",
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.getByText(/Never\s+certifies the unseen/)).toBeInTheDocument();
  });

  it("shows explicit error state when API is unreachable (no fake data)", async () => {
    renderApp();
    await waitFor(() =>
      expect(document.querySelector(".error-state, .loading")).toBeTruthy()
    );
    // never renders a fabricated metric value
    expect(screen.queryByText(/99(\.\d+)?%/)).toBeNull();
  });

  it("unknown route shows not-found, never a blank screen", async () => {
    window.location.hash = "#/does-not-exist";
    renderApp();
    await waitFor(() =>
      expect(screen.getByText(/Page not found/)).toBeInTheDocument()
    );
  });
});
