import type { ArbEntry } from "../types";

interface Props {
  arbLog: ArbEntry[];
  totalDetected: number;
}

function getTier(profit: number): "tier-high" | "tier-mid" | "tier-low" {
  if (profit > 10) return "tier-high";
  if (profit > 5)  return "tier-mid";
  return "tier-low";
}

function getTierLabel(profit: number): string {
  if (profit > 10) return "🟢";
  if (profit > 5)  return "🟡";
  return "🔵";
}

export function ArbFeed({ arbLog, totalDetected }: Props) {
  const recent = [...arbLog].reverse().slice(0, 20);

  return (
    <div className="arb-feed">
      <div className="feed-header">
        <span>Live Arb Opportunities</span>
        <span className="feed-count">{totalDetected.toLocaleString()} total</span>
      </div>

      <div className="feed-list">
        {recent.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📡</div>
            <div className="empty-text">No opportunities detected yet. Start the engine to begin scanning.</div>
          </div>
        ) : (
          recent.map((entry, i) => {
            const tier = getTier(entry.profit_bps);
            const pathStr = entry.path.join(" → ");
            return (
              <div key={`${entry.tick}-${i}`} className={`feed-entry ${tier}`}>
                <div className="feed-profit">
                  {getTierLabel(entry.profit_bps)} +{entry.profit_bps.toFixed(1)} bps
                </div>
                <div className="feed-path">{pathStr}</div>
                <div className="feed-meta">
                  tick #{entry.tick.toLocaleString()} · {entry.legs}-leg cycle
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
