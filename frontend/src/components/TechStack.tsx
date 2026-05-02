export function TechStack() {
  const cards = [
    {
      title: "Bellman-Ford  O(V·E)",
      desc: "Detects negative-weight cycles in the log-price graph. A cycle returning profit > 1x starting capital signals triangular arbitrage.",
      color: "green",
    },
    {
      title: "Dueling DQN",
      desc: "Separates value stream V(s) from advantage stream A(s,a). Learns WHEN to execute vs. hold, accounting for spread costs and latency.",
      color: "blue",
    },
    {
      title: "Double DQN",
      desc: "Uses the policy network to select actions but the target network to evaluate them. Reduces Q-value overestimation bias significantly.",
      color: "purple",
    },
    {
      title: "GARCH(1,1) Simulation",
      desc: "Simulates realistic volatility clustering (calm periods followed by bursts). Provides training data with microstructure realism.",
      color: "amber",
    },
  ];

  const stackBadges = [
    { label: "Python 3.11", color: "blue" },
    { label: "PyTorch 2.x", color: "amber" },
    { label: "FastAPI", color: "green" },
    { label: "WebSockets", color: "blue" },
    { label: "React 18", color: "blue" },
    { label: "D3.js", color: "purple" },
    { label: "Recharts", color: "green" },
    { label: "TypeScript", color: "blue" },
    { label: "Vite", color: "amber" },
    { label: "NumPy", color: "blue" },
  ];

  return (
    <div className="tech-section">
      <div className="tech-title">
        <span>⚙️</span>
        <span>How It Works</span>
        <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 400, marginLeft: 8 }}>
          — Architecture & Algorithms
        </span>
      </div>

      <div className="tech-grid">
        {cards.map(c => (
          <div key={c.title} className="tech-card">
            <div className="tech-card-title" style={{ color: `var(--accent-${c.color})` }}>
              {c.title}
            </div>
            <div className="tech-card-desc">{c.desc}</div>
          </div>
        ))}
      </div>

      <div style={{ height: 1, background: "var(--bg-border-dim)", margin: "16px 0" }} />

      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 10 }}>
        Tech Stack
      </div>
      <div className="stack-badges">
        {stackBadges.map(b => (
          <span key={b.label} className={`badge badge-${b.color}`}>{b.label}</span>
        ))}
      </div>
    </div>
  );
}
