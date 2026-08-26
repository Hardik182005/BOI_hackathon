import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "../landing.css";

/* ============================================================================
   MuleGuard · Trinetra — public landing page.

   Layout, motion and type system are the KLEOS public-site pattern; every word
   of content, and the pipeline in the hero, are this project's own. Nothing
   here reaches the analyst console: the console's routes, data layer and
   stylesheet are untouched, and this page renders no number the repository
   cannot show you the artifact for.
   ========================================================================== */

const NAV_LINKS = [
  { label: "Trinetra", id: "trinetra" },
  { label: "Guardrails", id: "guardrails" },
];

const MARQUEE = [
  "9,082 accounts", "3,925 raw columns", "81 confirmed mules", "0.8919% prevalence",
  "120 selected features", "Leakage firewall", "Nested CV · 5 outer × 3 repeats",
  "Platt calibration", "Mondrian conformal", "IsolationForest challenger",
  "SHAP evidence", "Append-only audit", "ProofGraph", "Cohort Radar",
  "No MCP", "No browser agents", "Fully offline",
];

/* ---------- icons: inline, so the page ships no icon dependency ----------- */
type IconProps = { size?: number };
const svg = (size: number) => ({
  width: size, height: size, viewBox: "0 0 24 24", fill: "none",
  stroke: "currentColor", strokeWidth: 1.7,
  strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
});
const Shield = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" /><path d="m9 12 2 2 4-4" /></svg>
);
const Eye = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" /></svg>
);
const Scan = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2" /><path d="M3 12h18" /></svg>
);
const Filter = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><path d="M3 4h18l-7 8v7l-4 2v-9L3 4Z" /></svg>
);
const Layers = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><path d="m12 2 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 17 9 5 9-5" /></svg>
);
const Users = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><path d="M16 20v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M22 20v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8" /></svg>
);
const Lock = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><rect x="4" y="10" width="16" height="11" rx="2" /><path d="M8 10V7a4 4 0 1 1 8 0v3" /></svg>
);
const Database = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></svg>
);
const Branch = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><circle cx="6" cy="5" r="2.5" /><circle cx="6" cy="19" r="2.5" /><circle cx="18" cy="12" r="2.5" /><path d="M6 7.5v9M8.5 19h4a3 3 0 0 0 3-3v-2M8.5 5h4a3 3 0 0 1 3 3v2" /></svg>
);
const Doc = ({ size = 18 }: IconProps) => (
  <svg {...svg(size)}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z" /><path d="M14 3v5h5M9 13h6M9 17h4" /></svg>
);
const Arrow = ({ size = 17 }: IconProps) => (
  <svg {...svg(size)} className="lp-arrow"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
);
const Chevron = ({ size = 15 }: IconProps) => (
  <svg {...svg(size)}><path d="m9 6 6 6-6 6" /></svg>
);
const Burger = ({ size = 24 }: IconProps) => (
  <svg {...svg(size)}><path d="M3 6h18M3 12h18M3 18h18" /></svg>
);
const Close = ({ size = 26 }: IconProps) => (
  <svg {...svg(size)}><path d="M18 6 6 18M6 6l12 12" /></svg>
);

/* ---------- motion preference, read from the stylesheet ------------------- */
function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  const probe = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = probe.current;
    if (!el) return;
    setReduced(getComputedStyle(el).getPropertyValue("--lp-reduced").trim() === "1");
  }, []);
  return {
    reduced,
    probe: <span ref={probe} className="lp-motion-probe" aria-hidden />,
  };
}

/* ---------- reveal-on-scroll: one observer for the whole page -------------- */
function useReveal(ready: boolean) {
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ready || !root.current) return;
    const targets = root.current.querySelectorAll(".lp-reveal, .lp-pipe-rail");
    if (!("IntersectionObserver" in globalThis)) {
      targets.forEach((t) => t.classList.add("is-in"));
      return;
    }
    const obs = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("is-in"); obs.unobserve(e.target); }
      }),
      { rootMargin: "-60px" },
    );
    targets.forEach((t) => obs.observe(t));
    return () => obs.disconnect();
  }, [ready]);
  return root;
}

/* ---------- animated counter ---------------------------------------------- */
function Counter({ to, decimals = 0, suffix = "", reduced }: {
  to: number; decimals?: number; suffix?: string; reduced: boolean;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [val, setVal] = useState(reduced ? to : 0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reduced || !("IntersectionObserver" in globalThis)) { setVal(to); return; }
    let raf = 0;
    const obs = new IntersectionObserver((entries) => {
      if (!entries[0].isIntersecting) return;
      obs.disconnect();
      const start = performance.now();
      const tick = (now: number) => {
        const p = Math.min(1, (now - start) / 1400);
        setVal((1 - Math.pow(1 - p, 3)) * to);
        if (p < 1) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }, { rootMargin: "-40px" });
    obs.observe(el);
    return () => { obs.disconnect(); cancelAnimationFrame(raf); };
  }, [to, reduced]);

  return <span ref={ref} className="lp-nums">{val.toFixed(decimals)}{suffix}</span>;
}

/* ---------- small primitives ----------------------------------------------- */
function Reveal({ children, delay = 0, className = "" }: {
  children: React.ReactNode; delay?: number; className?: string;
}) {
  return (
    <div className={`lp-reveal ${className}`} style={{ "--d": `${delay}ms` } as React.CSSProperties}>
      {children}
    </div>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return <span className="lp-eyebrow">{children}</span>;
}

function Words({ text, delay = 0, className = "" }: { text: string; delay?: number; className?: string }) {
  return (
    <span className={`lp-word ${className}`}>
      {text.split(" ").map((w, i) => (
        <span key={`${w}-${i}`}>
          <span style={{ "--d": `${delay + i * 80}ms` } as React.CSSProperties}>{w}&nbsp;</span>
        </span>
      ))}
    </span>
  );
}

/* ============================================================================
   THE GUARDRAIL PIPELINE — hero rail.

   This is the page's centrepiece and it is this project's actual scoring path:
   a request has to survive the schema gate, all three lenses and a rules-only
   policy engine before a human being is asked to look at it. The one number on
   it is marked as a walkthrough, because the real ones live in the console.
   ========================================================================== */
const STAGES = [
  {
    icon: Filter, title: "Schema & leakage gate", sub: "422 refusal · never zero-filled",
    chips: [
      { t: "F3924 → 422", k: "red" },
      { t: "missing feature → 422", k: "red" },
      { t: "no silent imputation", k: "" },
    ],
  },
  {
    icon: Eye, title: "Lens 1 · DETECT", sub: "XGBoost champion · 120 features",
    meter: true,
    chips: [
      { t: "Platt-calibrated", k: "" },
      { t: "LGBM + CatBoost agreement", k: "" },
    ],
  },
  {
    icon: Scan, title: "Lens 3 · Never certify the unseen", sub: "OOD detector · IsolationForest challenger",
    chips: [
      { t: "in distribution ✓", k: "green" },
      { t: "else → OOD_REVIEW", k: "violet" },
    ],
  },
  {
    icon: Shield, title: "Lens 2 · Spare the look-alike", sub: "hard-negative verifier · Mondrian conformal",
    chips: [
      { t: "survives verifier ✓", k: "green" },
      { t: "abstains → MONITOR", k: "green" },
    ],
  },
  {
    icon: Layers, title: "Policy engine", sub: "deterministic · no ML · no LLM", highlight: true,
    chips: [
      { t: "CRITICAL_REVIEW", k: "ink" },
      { t: "URGENT_REVIEW", k: "amber" },
      { t: "STANDARD_REVIEW", k: "blue" },
    ],
  },
  {
    icon: Users, title: "Human analyst", sub: "recommendation only · append-only audit",
    chips: [
      { t: "no automatic freeze", k: "green" },
      { t: "named analyst + 2nd approver", k: "" },
    ],
    branch: "Local LLM · narrates verified facts, cannot touch a score · 15/15 guardrail cases",
  },
];

function GuardrailPipeline({ reduced }: { reduced: boolean }) {
  const [active, setActive] = useState(0);
  const [risk, setRisk] = useState(reduced ? 0.94 : 0);
  const railRef = useRef<HTMLDivElement>(null);
  const [seen, setSeen] = useState(reduced);

  useEffect(() => {
    const el = railRef.current;
    if (!el || reduced || !("IntersectionObserver" in globalThis)) { setSeen(true); return; }
    const obs = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) { setSeen(true); obs.disconnect(); }
    }, { rootMargin: "-40px" });
    obs.observe(el);
    return () => obs.disconnect();
  }, [reduced]);

  useEffect(() => {
    if (!seen || reduced) return;
    let raf = 0;
    const begin = setTimeout(() => {
      const start = performance.now();
      const tick = (now: number) => {
        const p = Math.min(1, (now - start) / 1500);
        setRisk((1 - Math.pow(1 - p, 3)) * 0.94);
        if (p < 1) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }, 700);
    return () => { clearTimeout(begin); cancelAnimationFrame(raf); };
  }, [seen, reduced]);

  useEffect(() => {
    if (!seen || reduced) return;
    const id = setInterval(() => setActive((a) => (a + 1) % STAGES.length), 2400);
    return () => clearInterval(id);
  }, [seen, reduced]);

  return (
    <div className="lp-pipe">
      <div className="lp-pipe-badges">
        <span className="lp-chip lp-chip-green">
          <span className="lp-dot lp-dot-live"><span className="lp-dot" /></span>
          Guardrail pipeline · live
        </span>
        <span className="lp-chip">
          <span className="lp-dot" style={{ background: "#EF9D2B" }} />
          Illustrative walkthrough
        </span>
      </div>

      <div className="lp-pipe-rail" ref={railRef}>
        {!reduced && seen && (
          <span className="lp-pipe-packet" aria-hidden><i /><i /></span>
        )}

        {STAGES.map((s, i) => {
          const Icon = s.icon;
          const on = active === i;
          return (
            <div
              key={s.title}
              className={`lp-stage${on ? " is-active" : ""}${s.highlight ? " lp-stage-hl" : ""}`}
              onMouseEnter={() => setActive(i)}
            >
              <div className="lp-stage-node">
                {on && !reduced && <span className="lp-halo" aria-hidden />}
                <span><Icon size={20} /></span>
              </div>

              <div className="lp-stage-head">
                <span className="lp-stage-t">{s.title}</span>
                <Chevron size={14} />
              </div>
              <div className="lp-stage-s">{s.sub}</div>

              {s.meter && (
                <div className="lp-meter">
                  <div className="lp-meter-top">
                    <span className="lp-meter-k">Calibrated risk · example account</span>
                    <span className="lp-meter-v">{risk.toFixed(2)}<small>/1.00</small></span>
                  </div>
                  <div className="lp-meter-bar">
                    <span style={{ width: seen ? "94%" : "0%" }} />
                  </div>
                  <div className="lp-meter-note">
                    A probability, not a verdict. Nothing is frozen by this number.
                  </div>
                </div>
              )}

              <div className="lp-stage-body">
                {s.chips.map((c) => (
                  <span key={c.t} className={`lp-chip${c.k ? ` lp-chip-${c.k}` : ""}`}>{c.t}</span>
                ))}
              </div>

              {s.branch && (
                <div className="lp-branch">
                  <div className="lp-branch-t">{s.branch}</div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ============================================================================
   page content
   ========================================================================== */
const CHAIN = [
  { n: "01", t: "Immutable dataset", d: "9,082 × 3,925, SHA-256 pinned. The raw file is never modified and never committed." },
  { n: "02", t: "Leakage firewall", d: "F3924 · F3912 · F2230 · index quarantined. The build fails if one reaches a model." },
  { n: "03", t: "Sealed split", d: "Locked test of 1,818 held back and touched once; the 7,264-row dev set does all the work." },
  { n: "04", t: "Model core", d: "120 stability-selected features, nested CV of 5 outer folds × 3 repeats, calibrated." },
  { n: "05", t: "Lens 1 · Detect", d: "The calibrated probability, with LightGBM and CatBoost agreement recorded alongside it." },
  { n: "06", t: "Lens 2 · Spare", d: "Hard-negative verifier and Mondrian conformal abstention protect the look-alike." },
  { n: "07", t: "Lens 3 · Never certify", d: "OOD detection: benign means not currently flagged, and never means safe." },
  { n: "08", t: "Policy engine", d: "Rules only — no ML, no LLM — assigning one of five review tiers to a human." },
];

const GUARDRAILS = [
  { icon: Filter, n: "G.01", t: "No leakage reaches a model", d: "Four columns are quarantined at the firewall, and the test suite fails the build if any of them enters a model frame." },
  { icon: Lock, n: "G.02", t: "The locked test is touched once", d: "A sentinel log records the single evaluation. A second run refuses to execute rather than quietly overwrite the number." },
  { icon: Database, n: "G.03", t: "Natural prevalence, everywhere", d: "No resampling in the accepted model — backed by a seven-arm ablation rather than by assertion." },
  { icon: Users, n: "G.04", t: "Protected attributes excluded, and priced", d: "Gender never reaches the frame. An eight-arm ablation prices the exclusion at −0.0001 AP and shows the feature also hurts." },
  { icon: Shield, n: "G.05", t: "Nothing freezes itself", d: "Every tier is a recommendation. A freeze needs a named analyst plus a second approver, and the whole chain is audited." },
  { icon: Branch, n: "G.06", t: "The narrator cannot touch a score", d: "The scoring path contains no model call at all. Narration is schema-validated and discarded whole on any deviation." },
  { icon: Doc, n: "G.07", t: "Every number traces to a file", d: "Displayed metrics are recomputed from saved prediction files and asserted at gate time, not typed into a slide." },
];

export default function Landing() {
  const { reduced, probe } = useReducedMotion();
  const [scrolled, setScrolled] = useState(false);
  const [menu, setMenu] = useState(false);
  const [active, setActive] = useState("");
  const root = useReveal(true);

  useEffect(() => {
    const onScroll = () => setScrolled(globalThis.scrollY > 40);
    onScroll();
    globalThis.addEventListener("scroll", onScroll, { passive: true });
    return () => globalThis.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!("IntersectionObserver" in globalThis)) return;
    const obs = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) setActive(e.target.id); }),
      { rootMargin: "-45% 0px -50% 0px" },
    );
    NAV_LINKS.forEach((l) => {
      const el = document.getElementById(l.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    document.body.style.overflow = menu ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [menu]);

  return (
    <div className="lp" ref={root}>
      {probe}
      <div className="lp-grain" aria-hidden />

      {/* ================= NAV ================= */}
      <nav className={`lp-nav${scrolled ? " is-scrolled" : ""}`}>
        <div className="lp-wrap lp-nav-inner">
          <Link to="/" className="lp-mark"><i />MuleGuard</Link>

          <div className="lp-nav-links">
            {NAV_LINKS.map((l) => (
              <a
                key={l.id}
                href={`#${l.id}`}
                className={`lp-nav-link${active === l.id ? " is-active" : ""}`}
              >
                {l.label}
              </a>
            ))}
            <Link to="/model-card" className="lp-nav-link">Model card</Link>
            <Link to="/overview" className="lp-nav-cta"><span>Analyst console</span></Link>
          </div>

          <button className="lp-burger" aria-label="Open menu" aria-expanded={menu} onClick={() => setMenu(true)}>
            <Burger />
          </button>
        </div>
      </nav>

      {menu && (
        <div className="lp-sheet">
          <div className="lp-sheet-top">
            <span className="lp-mark"><i />MuleGuard</span>
            <button aria-label="Close menu" onClick={() => setMenu(false)}><Close /></button>
          </div>
          <div className="lp-sheet-body">
            {NAV_LINKS.map((l) => (
              <a key={l.id} href={`#${l.id}`} onClick={() => setMenu(false)}>{l.label}</a>
            ))}
            <Link to="/model-card" onClick={() => setMenu(false)}>Model card</Link>
            <Link to="/overview" className="lp-btn lp-btn-ghost" onClick={() => setMenu(false)}>
              Analyst console <Arrow />
            </Link>
          </div>
        </div>
      )}

      {/* ================= HERO ================= */}
      <section className="lp-hero">
        <div className="lp-hero-grid" aria-hidden />
        <div className="lp-hero-glow" aria-hidden />

        <div className="lp-wrap lp-hero-inner">
          <div>
            <Reveal delay={40}>
              <p className="lp-hero-kicker">Sees the mule. Spares the look-alike.</p>
            </Reveal>

            <h1>
              <Words text="Catch the mule." delay={60} />
              <Words text="Spare the look-alike." delay={340} className="lp-stroke" />
            </h1>

            <Reveal delay={620}>
              <p className="lp-lead lp-hero-lead">
                MuleGuard · Trinetra classifies suspicious mule accounts at{" "}
                <b>0.89% prevalence</b> — three lenses, a rules-only policy engine, and
                guardrails that make the wrong answer structurally impossible rather than
                merely discouraged.
              </p>
            </Reveal>

            <Reveal delay={700}>
              <div className="lp-hero-actions">
                <Link to="/overview" className="lp-btn lp-btn-solid">
                  Open the analyst console <Arrow />
                </Link>
                <a href="#guardrails" className="lp-btn lp-btn-ghost">
                  <Shield size={17} /> See the guardrails
                </a>
              </div>
            </Reveal>
          </div>

          <Reveal delay={420}>
            <GuardrailPipeline reduced={reduced} />
          </Reveal>
        </div>

        <div className="lp-marquee" aria-hidden>
          <div className="lp-marquee-track">
            {[...MARQUEE, ...MARQUEE].map((m, i) => <span key={`${m}-${i}`}>{m}</span>)}
          </div>
        </div>
      </section>

      {/* ================= STATS ================= */}
      <section className="lp-stats">
        <div className="lp-stats-grid">
          {[
            { to: 0.726, d: 3, s: "", l: "PR-AUC on the sealed locked test, touched exactly once" },
            { to: 77.7, d: 1, s: "×", l: "lift over a 0.89% base rate, at the same split" },
            { to: 0.0015, d: 4, s: "", l: "expected calibration error — the probability means what it says" },
            { to: 0, d: 0, s: "", l: "accounts this system has ever frozen without a named analyst" },
          ].map((st, i) => (
            <Reveal key={st.l} delay={i * 80} className="lp-stat">
              <div className="lp-stat-n">
                <Counter to={st.to} decimals={st.d} suffix={st.s} reduced={reduced} />
              </div>
              <div className="lp-stat-l">{st.l}</div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ================= 01 · TRINETRA (dark) ================= */}
      <section id="trinetra" className="lp-section lp-dark">
        <div className="lp-wrap">
          <div style={{ maxWidth: 860, margin: "0 auto 56px", textAlign: "center" }}>
            <Reveal><div style={{ display: "flex", justifyContent: "center" }}><Eyebrow>The solution · 01</Eyebrow></div></Reveal>
            <Reveal delay={50}>
              <h2 className="lp-h2" style={{ marginTop: 24 }}>
                Three eyes, one <span className="lp-stroke">audited chain.</span>
              </h2>
            </Reveal>
            <Reveal delay={100}>
              <p className="lp-lead" style={{ maxWidth: 620, margin: "24px auto 0" }}>
                Trinetra is not three models voting. It is three distinct failure modes, each
                with its own dedicated defence, composed in sequence and resolved by a policy
                engine that contains no ML and no language model at all.
              </p>
            </Reveal>
          </div>

          <div className="lp-chain">
            {CHAIN.map((s, i) => (
              <Reveal key={s.n} delay={i * 50}>
                <div className="lp-chain-card">
                  <div className="lp-chain-top">
                    <span className="lp-chain-n">{s.n}</span>
                    {i < CHAIN.length - 1 && <Chevron size={15} />}
                  </div>
                  <div className="lp-chain-t">{s.t}</div>
                  <p className="lp-chain-d">{s.d}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ================= 02 · GUARDRAILS ================= */}
      <section id="guardrails" className="lp-section lp-line-top">
        <div className="lp-wrap">
          <Reveal><span className="lp-numeral">02</span></Reveal>
          <div style={{ marginTop: 28 }}><Reveal><Eyebrow>Seven machine-enforced guarantees</Eyebrow></Reveal></div>
          <Reveal delay={50}>
            <h2 className="lp-h3" style={{ maxWidth: 720, margin: "24px 0 56px" }}>
              Guardrails are not promises. Each one fails the build.
            </h2>
          </Reveal>

          <div className="lp-grid lp-grid-2 lp-grid-3">
            {GUARDRAILS.map((f, i) => (
              <Reveal key={f.n} delay={i * 50}>
                <div className="lp-card" style={{ display: "flex", flexDirection: "column" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span className="lp-icon lp-icon-lg"><f.icon size={18} /></span>
                    <span className="lp-mono" style={{ fontSize: 12, letterSpacing: "0.2em", color: "#8A93A6" }}>{f.n}</span>
                  </div>
                  <h3 style={{ marginTop: 24, fontSize: 20, fontWeight: 500, lineHeight: 1.18, letterSpacing: "-0.02em" }}>{f.t}</h3>
                  <p className="lp-card-body">{f.d}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ================= FOOTER ================= */}
      <footer className="lp-footer">
        <div className="lp-wrap">
          <div className="lp-footer-top">
            <Link to="/" className="lp-mark"><i />MuleGuard</Link>
            <div className="lp-footer-meta">PSB Cybersecurity, Fraud &amp; AI Hackathon 2026 · PS2</div>
            <div className="lp-footer-meta">Team Kryptonite</div>
          </div>
          <div className="lp-footer-word">MULEGUARD</div>
          <div className="lp-footer-fine">
            <span>Hardik Hinduja · Avinash Gehi · Sahil Deshmukh · Siddharth Dey</span>
            <span>© 2026 — behavioural risk is not proof of criminal intent · no MCP, no browser agents, no internet</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
