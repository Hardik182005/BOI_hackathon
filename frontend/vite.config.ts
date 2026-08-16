import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://127.0.0.1:8001",
      "/health": "http://127.0.0.1:8001",
      // Graph Lab asks the backend which routes it actually serves rather than
      // assuming a counterparty-graph endpoint is deployed. Without this entry
      // the dev server answers /openapi.json with index.html and the page
      // reports "the response was not JSON".
      "/openapi.json": "http://127.0.0.1:8001",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    globals: true,
    // vitest stubs stylesheets to nothing by default, which also empties a
    // `?raw` import of one. The resolution test reads styles.css as text - it
    // is the only place the desktop layout is actually defined - so the stub
    // has to be off for that import to carry the rules it asserts on.
    css: true,
  },
} as any);
