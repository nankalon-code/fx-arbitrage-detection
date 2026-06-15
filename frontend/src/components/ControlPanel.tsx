import type { EngineConfig } from "../types";

interface Props {
  connected: boolean;
  running: boolean;
  config: EngineConfig;
  onStart: () => void;
  onStop: () => void;
  onConfigChange: (updates: Partial<EngineConfig>) => void;
  isOpen?: boolean;
}

export function ControlPanel({ connected, running, config, onStart, onStop, onConfigChange, isOpen }: Props) {
  return (
    <aside className={`sidebar${isOpen ? " open" : ""}`}>
      <div className="sidebar-section">
        <span className="sidebar-label">Engine Control</span>
        <button
          className="btn btn-primary"
          onClick={onStart}
          disabled={!connected || running}
        >
          Start Engine
        </button>
        <button
          className="btn btn-danger"
          onClick={onStop}
          disabled={!running}
        >
          Stop Engine
        </button>
      </div>

      <div className="divider" />

      <div className="sidebar-section">
        <span className="sidebar-label">Parameters</span>

        <div className="slider-group">
          <div className="slider-header">
            <span>Currency Pairs</span>
            <span className="slider-value">{config.n_pairs}</span>
          </div>
          <input
            type="range" min={4} max={20} step={1}
            value={config.n_pairs}
            onChange={e => onConfigChange({ n_pairs: +e.target.value })}
            disabled={running}
          />
        </div>

        <div className="slider-group">
          <div className="slider-header">
            <span>Min Profit (bps)</span>
            <span className="slider-value">{config.min_profit_bps}</span>
          </div>
          <input
            type="range" min={1} max={20} step={1}
            value={config.min_profit_bps}
            onChange={e => onConfigChange({ min_profit_bps: +e.target.value })}
            disabled={running}
          />
        </div>

        <div className="slider-group">
          <div className="slider-header">
            <span>Arb Injection Prob</span>
            <span className="slider-value">{(config.arb_prob * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range" min={1} max={20} step={1}
            value={Math.round(config.arb_prob * 100)}
            onChange={e => onConfigChange({ arb_prob: +e.target.value / 100 })}
            disabled={running}
          />
        </div>

        <div className="slider-group">
          <div className="slider-header">
            <span>Speed (ticks/s)</span>
            <span className="slider-value">{config.speed}</span>
          </div>
          <input
            type="range" min={1} max={50} step={1}
            value={config.speed}
            onChange={e => onConfigChange({ speed: +e.target.value })}
          />
        </div>
      </div>

      <div className="divider" />

      <div className="sidebar-section">
        <span className="sidebar-label">Architecture</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {[
            { label: "Bellman-Ford", sub: "O(V·E) negative cycle detection" },
            { label: "Dueling DQN", sub: "V(s) + A(s,a) dual stream" },
            { label: "Double DQN", sub: "Reduces Q-value overestimation" },
            { label: "GARCH(1,1)", sub: "Volatility clustering realism" },
            { label: "Kafka Pipeline", sub: "50K+ ticks/sec ingestion" },
            { label: "Redis Cache", sub: "Sub-ms state lookups" },
            { label: "FastAPI WS", sub: "Real-time async streaming" },
          ].map(s => (
            <div key={s.label} className="sidebar-stack-item">
              <div className="sidebar-stack-label">{s.label}</div>
              <div className="sidebar-stack-sub">{s.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
