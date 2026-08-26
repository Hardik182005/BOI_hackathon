import { Link, NavLink, Route, Routes } from "react-router-dom";
import Landing from "./pages/Landing";
import Overview from "./pages/Overview";
import AlertQueue from "./pages/AlertQueue";
import CaseDetail from "./pages/CaseDetail";
import ModelPerformance from "./pages/ModelPerformance";
import FeatureIntelligence from "./pages/FeatureIntelligence";
import DriftMonitoring from "./pages/DriftMonitoring";
import ModelCard from "./pages/ModelCard";
import ValidationLab from "./pages/ValidationLab";
import BusinessValue from "./pages/BusinessValue";
import CapacityOptimizer from "./pages/CapacityOptimizer";
import GraphLab from "./pages/GraphLab";
import ProofGraph from "./pages/ProofGraph";

const pages = [
  { to: "/overview", label: "Executive Overview", el: <Overview /> },
  { to: "/queue", label: "Alert Queue", el: <AlertQueue /> },
  { to: "/performance", label: "Model Performance", el: <ModelPerformance /> },
  { to: "/features", label: "Feature Intelligence", el: <FeatureIntelligence /> },
  { to: "/drift", label: "Drift & Monitoring", el: <DriftMonitoring /> },
  { to: "/model-card", label: "Model Card", el: <ModelCard /> },
  { to: "/validation", label: "Test Model / Validation Lab", el: <ValidationLab /> },
  { to: "/value", label: "Business Value", el: <BusinessValue /> },
  { to: "/capacity", label: "Analyst Capacity", el: <CapacityOptimizer /> },
  { to: "/graph", label: "Graph Lab", el: <GraphLab /> },
];

// The analyst console. Every route except the public landing page renders
// inside it, so the sidebar, its ten links and the `.main` column are the same
// on all of them - which is what the resolution suite asserts per route.
function Console({ children }: { children: React.ReactNode }) {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          {/* The wordmark is the way back out to the landing page; it sits in
              `.brand` rather than `.nav` so the navigation stays ten links. */}
          <Link to="/" className="brand-home">
            <h1>MuleGuard</h1>
          </Link>
          <div className="tagline">
            the Trinetra engine · Sees the mule. Spares the look-alike. Never
            certifies the unseen.
          </div>
        </div>
        <nav className="nav">
          {pages.map((p) => (
            <NavLink key={p.to} to={p.to} end>
              {p.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      {pages.map((p) => (
        <Route key={p.to} path={p.to} element={<Console>{p.el}</Console>} />
      ))}
      <Route path="/cases/:caseId" element={<Console><CaseDetail /></Console>} />
      <Route path="/proof/:caseId" element={<Console><ProofGraph /></Console>} />
      <Route
        path="*"
        element={<Console><div className="empty">Page not found - use the navigation.</div></Console>}
      />
    </Routes>
  );
}
