import { useState, useEffect } from "react";
import type { ConnectionStatus } from "../types";

interface Props {
  connected: boolean;
  running: boolean;
  connectionStatus: ConnectionStatus;
  uptime?: number;
  onMenuToggle?: () => void;
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function HeroBanner({ connected, running, connectionStatus, uptime, onMenuToggle }: Props) {
  const [clock, setClock] = useState("");

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setClock(now.toLocaleTimeString("en-US", { hour12: false }));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const statusLabel = connectionStatus === "connecting" ? "Connecting"
    : !connected ? "Disconnected"
    : running ? "Live" : "Standby";

  const statusClass = connectionStatus === "connecting" ? "connecting"
    : !connected ? "disconnected"
    : running ? "running" : "stopped";

  return (
    <header className="hero">
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button className="mobile-menu-btn" onClick={onMenuToggle} aria-label="Toggle menu">
          ☰
        </button>
        <div className="hero-left">
          <h1 className="hero-title">FX Arbitrage Detection Engine</h1>
          <p className="hero-subtitle">
            Bellman-Ford Cycle Detection · Deep Q-Networks · 20 Currency Pairs
          </p>
        </div>
      </div>

      <div className="hero-right">
        <div className="hero-badges">
          <span className="badge badge-green">Bellman-Ford</span>
          <span className="badge badge-blue">Dueling DQN</span>
          <span className="badge badge-amber">PyTorch</span>
          <span className="badge badge-purple">FastAPI</span>
          <span className="badge badge-cyan">React</span>
        </div>

        {running && uptime != null && (
          <div className="uptime-clock">
            <span className="uptime-val">{formatUptime(uptime)}</span>
          </div>
        )}

        <div className="uptime-clock">
          {clock}
        </div>

        <div className={`status-badge ${statusClass}`}>
          <div className="pulse-dot" />
          {statusLabel}
        </div>
      </div>
    </header>
  );
}
