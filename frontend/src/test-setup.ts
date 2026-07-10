import "@testing-library/jest-dom";

// Recharts needs ResizeObserver in jsdom
class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as any).ResizeObserver = (globalThis as any).ResizeObserver ?? RO;
