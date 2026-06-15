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

export function ArbFeed({ arbLog, totalDetected }: Props) {
  const recent = [...arbLog].reverse().slice(0, 20);

  return (
    <div className="arb-feed">
      <div className="feed-header">
        <span>
          {totalDetected > 0 && <span className="live-dot" />}
          Live Arb Opportunities
        </span>
        <span className="feed-count">{totalDetected.toLocaleString()} total</span>
      </div>

      <div className="feed-list">
        {recent.length === 0 ? (
          <div className="empty-state">
            <div className="empty-text">No opportunities detected yet. Start the engine to begin scanning.</div>
          </div>
        ) : (
          recent.map((entry, i) => {
            const tier = getTier(entry.profit_bps ?? 0);
            const pathStr = Array.isArray(entry.path) ? entry.path.join(" → ") : "—";
            return (
              <div key={`${entry.tick}-${i}`} className={`feed-entry ${tier}`}>
                <div className="feed-profit">
                  +{(entry.profit_bps ?? 0).toFixed(1)} bps
                </div>
                <div className="feed-path">{pathStr}</div>
                <div className="feed-meta">
                  tick #{(entry.tick ?? 0).toLocaleString()} · {entry.legs ?? 0}-leg cycle
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
