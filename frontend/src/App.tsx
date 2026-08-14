import { NavLink, Route, Routes } from "react-router-dom";
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
  { to: "/", label: "Executive Overview", el: <Overview /> },
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

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <h1>MuleGuard</h1>
          <div className="tagline">
            the Trinetra engine · Sees the mule. Spares the look-alike. Never
            certifies the unseen.
          </div>
        </div>
        <nav className="nav">
          {pages.map((p) => (
            <NavLink key={p.to} to={p.to} end={p.to === "/"}>
              {p.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Routes>
          {pages.map((p) => (
            <Route key={p.to} path={p.to} element={p.el} />
          ))}
          <Route path="/cases/:caseId" element={<CaseDetail />} />
          <Route path="/proof/:caseId" element={<ProofGraph />} />
          <Route
            path="*"
            element={<div className="empty">Page not found - use the navigation.</div>}
          />
        </Routes>
      </main>
    </div>
  );
}
