interface Props {
  connected: boolean;
  running: boolean;
}

export function HeroBanner({ connected, running }: Props) {
  const statusLabel = !connected ? "Disconnected" : running ? "Live" : "Standby";
  const statusClass = !connected ? "disconnected" : running ? "running" : "stopped";

  return (
    <header className="hero">
      <div className="hero-left">
        <h1 className="hero-title">FX Arbitrage Detection Engine</h1>
        <p className="hero-subtitle">
          Bellman-Ford Graph Algorithm · Dueling Double DQN · GARCH Volatility Simulation · Real-Time WebSocket Streaming
        </p>
      </div>

      <div className="hero-badges">
        <span className="badge badge-green">Bellman-Ford</span>
        <span className="badge badge-blue">Dueling DQN</span>
        <span className="badge badge-amber">PyTorch</span>
        <span className="badge badge-purple">FastAPI</span>
        <span className="badge badge-blue">React</span>

        <div className={`status-badge ${statusClass}`}>
          <div className="pulse-dot" />
          {statusLabel}
        </div>
      </div>
    </header>
  );
}
