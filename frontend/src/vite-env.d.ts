/// <reference types="vite/client" />

// Vite's ambient module types, which is what makes `?raw` imports typed. The
// resolution test reads styles.css as text to check the declared breakpoints
// and fixed widths - the stylesheet is never applied in jsdom, so the rules
// themselves are the only evidence available to it.
