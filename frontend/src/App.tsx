import { useState } from "react";
import "./index.css";
import { useWebSocket }    from "./hooks/useWebSocket";
import { ParticleBackground } from "./components/ParticleBackground";
import { HeroBanner }      from "./components/HeroBanner";
import { ControlPanel }    from "./components/ControlPanel";
import { MetricCard }      from "./components/MetricCard";
import { PnLChart }        from "./components/PnLChart";
import { PriceChart }      from "./components/PriceChart";
import { LossChart }       from "./components/LossChart";
import { GlobeNetwork }    from "./components/GlobeNetwork";
import { NetworkGraph }    from "./components/NetworkGraph";
import { ArbFeed }         from "./components/ArbFeed";
import { AgentStats }      from "./components/AgentStats";

const TABS = ["3D Globe", "PnL Curve", "Price Feed", "2D Network", "Loss Curve"] as const;
type Tab = typeof TABS[number];

export default function App() {
  const {
    connected, connectionStatus, running, lastFrame, config, error,
    startEngine, stopEngine, updateConfig, clearError,
  } = useWebSocket();

  const [activeTab, setActiveTab] = useState<Tab>("3D Globe");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const metrics = lastFrame?.metrics ?? null;
  const pnl = metrics?.total_pnl ?? 0;
  const isLive = running && connected;

  return (
    <>
      <ParticleBackground />

      {/* D3 tooltip */}
      <div
        className="d3-tooltip"
        style={{
          display: "none",
          position: "fixed",
          zIndex: 9999,
          background: "#ffffff",
          border: "1px solid var(--bg-border)",
          borderRadius: 4,
          padding: "8px 12px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--text-primary)",
          pointerEvents: "none",
          maxWidth: 220,
          boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
        }}
      />

      <div className="app">
        <HeroBanner
          connected={connected}
          running={running}
          connectionStatus={connectionStatus}
          uptime={metrics?.uptime}
          onMenuToggle={() => setSidebarOpen(prev => !prev)}
        />

        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button onClick={clearError}>×</button>
          </div>
        )}

        <ControlPanel
          connected={connected}
          running={running}
          config={config}
          onStart={startEngine}
          onStop={stopEngine}
          onConfigChange={updateConfig}
          isOpen={sidebarOpen}
        />

        <main className="main-content">

          {/* Metric Cards */}
          <div className="metrics-row">
            <MetricCard
              label="Ticks Processed"
              value={(metrics?.tick_count ?? 0).toLocaleString()}
              colorClass="blue"
              sub="market data points"
              isLive={isLive}
            />
            <MetricCard
              label="Arb Detected"
              value={(metrics?.arb_detected ?? 0).toLocaleString()}
              colorClass="green"
              sub="opportunities found"
              isLive={isLive && (metrics?.arb_detected ?? 0) > 0}
            />
            <MetricCard
              label="Total PnL"
              value={`${pnl >= 0 ? "+" : ""}${pnl.toFixed(1)} bps`}
              colorClass={pnl >= 0 ? "green" : "red"}
              positive={pnl > 0}
              negative={pnl < 0}
              sub="net after spread cost"
            />
            <MetricCard
              label="Trades"
              value={metrics?.trades ?? 0}
              colorClass="purple"
              sub="executions made"
            />
            <MetricCard
              label="Win Rate"
              value={`${(metrics?.win_rate ?? 0).toFixed(1)}%`}
              colorClass={(metrics?.win_rate ?? 0) >= 50 ? "green" : "amber"}
              sub="profitable trades"
            />
            <MetricCard
              label="Epsilon"
              value={(metrics?.epsilon ?? 1).toFixed(4)}
              colorClass="amber"
              sub="exploration rate"
            />
          </div>

          {/* Dashboard grid: charts + feed */}
          <div className="dashboard-grid">
            <div className="tabs">
              <div className="tab-list">
                {TABS.map(tab => (
                  <button
                    key={tab}
                    className={`tab-btn ${activeTab === tab ? "active" : ""}`}
                    onClick={() => setActiveTab(tab)}
                  >
                    {tab}
                  </button>
                ))}
              </div>
              <div className="tab-panel" style={activeTab === "3D Globe" ? { padding: 0, minHeight: 420 } : undefined}>
                {activeTab === "3D Globe" && (
                  <GlobeNetwork
                    opportunities={lastFrame?.opportunities ?? []}
                    prices={lastFrame?.prices ?? {}}
                    tickCount={metrics?.tick_count ?? 0}
                  />
                )}
                {activeTab === "PnL Curve" && (
                  <PnLChart pnlHistory={lastFrame?.pnl_history ?? []} />
                )}
                {activeTab === "Price Feed" && (
                  <PriceChart priceChart={lastFrame?.price_chart ?? {}} />
                )}
                {activeTab === "2D Network" && (
                  <NetworkGraph
                    opportunities={lastFrame?.opportunities ?? []}
                    prices={lastFrame?.prices ?? {}}
                  />
                )}
                {activeTab === "Loss Curve" && (
                  <LossChart lossHistory={lastFrame?.loss_history ?? []} />
                )}
              </div>
            </div>

            <div className="glass-card" style={{ padding: 16 }}>
              <ArbFeed
                arbLog={lastFrame?.arb_log ?? []}
                totalDetected={metrics?.arb_detected ?? 0}
              />
            </div>
          </div>

          {/* Agent Stats */}
          <div className="glass-card" style={{ padding: 20 }}>
            <AgentStats
              stats={lastFrame?.agent_stats ?? null}
              metrics={metrics}
              action={lastFrame?.action ?? "HOLD"}
              position={lastFrame?.position ?? false}
            />
          </div>

        </main>
      </div>
    </>
  );
}
