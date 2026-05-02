import type { AgentStats as AgentStatsType, Metrics } from "../types";

interface Props {
  stats: AgentStatsType | null;
  metrics: Metrics | null;
  action: string;
  position: boolean;
}

function StatCard({ label, value, color = "blue" }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="agent-stat">
      <div className="agent-stat-label">{label}</div>
      <div className="agent-stat-value" style={{ color: `var(--accent-${color})` }}>
        {value}
      </div>
    </div>
  );
}

export function AgentStats({ stats, metrics, action, position }: Props) {
  if (!stats || !metrics) {
    return (
      <div className="empty-state" style={{ height: 80 }}>
        <div className="empty-text">Agent stats will appear after training begins.</div>
      </div>
    );
  }

  const actionColor = action === "EXECUTE" ? "green" : action === "CLOSE" ? "amber" : "blue";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", color: "var(--text-muted)" }}>
          DQN Agent
        </span>
        <span className={`badge badge-${actionColor}`} style={{ marginLeft: "auto" }}>
          {action}
        </span>
        <span className={`badge badge-${position ? "green" : "blue"}`}>
          {position ? "IN POSITION" : "FLAT"}
        </span>
      </div>

      <div className="agent-stats-grid">
        <StatCard label="Steps" value={stats.steps?.toLocaleString() ?? "–"} color="blue" />
        <StatCard label="Epsilon (ε)" value={stats.epsilon?.toFixed(4) ?? "–"} color="amber" />
        <StatCard label="Buffer Size" value={stats.buffer_size?.toLocaleString() ?? "–"} color="purple" />
        <StatCard label="Mean Reward" value={stats.mean_reward_500?.toFixed(4) ?? "–"} color={stats.mean_reward_500 >= 0 ? "green" : "red"} />
        <StatCard label="Huber Loss" value={stats.mean_loss_100?.toFixed(6) ?? "–"} color="red" />
        <StatCard label="Total Reward" value={stats.total_reward?.toFixed(2) ?? "–"} color={stats.total_reward >= 0 ? "green" : "red"} />
        <StatCard label="Sharpe (500)" value={stats.sharpe_500?.toFixed(2) ?? "–"} color="blue" />
        <StatCard label="Win Rate" value={`${metrics.win_rate?.toFixed(1) ?? "0"}%`} color={metrics.win_rate >= 50 ? "green" : "amber"} />
      </div>
    </div>
  );
}
