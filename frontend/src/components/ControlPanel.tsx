import type { EngineConfig } from "../types";

interface Props {
  connected: boolean;
  running: boolean;
  config: EngineConfig;
  onStart: () => void;
  onStop: () => void;
  onConfigChange: (updates: Partial<EngineConfig>) => void;
}

export function ControlPanel({ connected, running, config, onStart, onStop, onConfigChange }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <span className="sidebar-label">Engine Control</span>
        <button
          className="btn btn-primary"
          onClick={onStart}
          disabled={!connected || running}
        >
          <span>▶</span> Start Engine
        </button>
        <button
          className="btn btn-danger"
          onClick={onStop}
          disabled={!running}
        >
          <span>■</span> Stop
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
            type="range" min={6} max={12} step={1}
            value={config.n_pairs}
            onChange={e => onConfigChange({ n_pairs: +e.target.value })}
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
        <span className="sidebar-label">Stack</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {[
            { label: "Bellman-Ford", sub: "O(V·E) cycle detection" },
            { label: "Dueling DQN", sub: "V(s) + A(s,a) streams" },
            { label: "Double DQN", sub: "Reduces overestimation" },
            { label: "GARCH Sim", sub: "Vol clustering realism" },
            { label: "FastAPI WS", sub: "Real-time streaming" },
          ].map(s => (
            <div key={s.label} style={{
              padding: "8px 10px",
              borderRadius: 6,
              background: "rgba(10,22,40,0.6)",
              border: "1px solid var(--bg-border-dim)"
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--accent-blue)" }}>{s.label}</div>
              <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2 }}>{s.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
