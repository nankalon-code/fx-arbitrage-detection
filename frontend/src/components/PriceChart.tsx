import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { useMemo } from "react";

interface Props {
  priceChart: Record<string, number[]>;
}

const COLORS = ["#4fc3f7", "#00ff88", "#ffb300", "#a78bfa", "#ff4d6d", "#22d3ee"];

export function PriceChart({ priceChart }: Props) {
  const pairs = useMemo(
    () => Object.keys(priceChart).filter(p => (priceChart[p]?.length ?? 0) > 1),
    [priceChart]
  );

  const data = useMemo(() => {
    if (pairs.length === 0) return [];
    const maxLen = Math.max(...pairs.map(p => priceChart[p].length));
    return Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, number | null> = { tick: i };
      pairs.forEach(p => {
        point[p] = priceChart[p]?.[i] ?? null;
      });
      return point;
    });
  }, [priceChart, pairs]);

  if (pairs.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-text">Price feeds will appear once the engine starts.</div>
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: 280 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,42,74,0.5)" />
          <XAxis dataKey="tick" hide />
          <YAxis
            tickFormatter={v => `${v.toFixed(2)}%`}
            tick={{ fill: "#475569", fontSize: 10, fontFamily: "JetBrains Mono" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(10,22,40,0.95)",
              border: "1px solid var(--bg-border)",
              borderRadius: 6,
              fontFamily: "JetBrains Mono",
              fontSize: 11,
            }}
            labelStyle={{ color: "#475569" }}
            formatter={(v: any) => [`${Number(v).toFixed(3)}%`]}
          />
          <Legend
            wrapperStyle={{ fontSize: 10, fontFamily: "JetBrains Mono", paddingTop: 8 }}
            iconType="plainline"
          />
          {pairs.slice(0, 6).map((pair, i) => (
            <Line
              key={pair}
              type="monotone"
              dataKey={pair}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={1.5}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
