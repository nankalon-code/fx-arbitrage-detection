import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

interface Props {
  pnlHistory: number[];
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.length) {
    const val = payload[0].value;
    return (
      <div style={{
        background: "rgba(10,22,40,0.95)",
        border: `1px solid ${val >= 0 ? "rgba(0,255,136,0.4)" : "rgba(255,77,109,0.4)"}`,
        borderRadius: 6,
        padding: "6px 12px",
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        color: val >= 0 ? "var(--accent-green)" : "var(--accent-red)",
      }}>
        {val >= 0 ? "+" : ""}{val.toFixed(2)} bps
      </div>
    );
  }
  return null;
};

export function PnLChart({ pnlHistory }: Props) {
  if (pnlHistory.length < 2) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📈</div>
        <div className="empty-text">PnL curve will appear once the engine starts trading.</div>
      </div>
    );
  }

  const data = pnlHistory.map((v, i) => ({ tick: i, pnl: v }));
  const latest = pnlHistory[pnlHistory.length - 1];
  const isPositive = latest >= 0;
  const color = isPositive ? "#00ff88" : "#ff4d6d";

  return (
    <div style={{ width: "100%", height: 280 }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={color} stopOpacity={0.2} />
              <stop offset="95%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,42,74,0.5)" />
          <XAxis dataKey="tick" hide />
          <YAxis
            tickFormatter={v => `${v.toFixed(0)}`}
            tick={{ fill: "#475569", fontSize: 10, fontFamily: "JetBrains Mono" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="pnl"
            stroke={color}
            strokeWidth={2}
            fill="url(#pnlGrad)"
            dot={false}
            animationDuration={200}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
