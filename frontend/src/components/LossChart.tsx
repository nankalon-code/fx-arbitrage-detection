import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

interface Props {
  lossHistory: number[];
}

export function LossChart({ lossHistory }: Props) {
  if (lossHistory.length < 2) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🧠</div>
        <div className="empty-text">DQN training loss will appear once the replay buffer fills (64 samples).</div>
      </div>
    );
  }

  const data = lossHistory.map((v, i) => ({ step: i, loss: v }));

  return (
    <div style={{ width: "100%", height: 280 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#ff4d6d" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#ff4d6d" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,42,74,0.5)" />
          <XAxis dataKey="step" hide />
          <YAxis
            tickFormatter={v => v.toFixed(4)}
            tick={{ fill: "#475569", fontSize: 10, fontFamily: "JetBrains Mono" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(10,22,40,0.95)",
              border: "1px solid rgba(255,77,109,0.4)",
              borderRadius: 6,
              fontFamily: "JetBrains Mono",
              fontSize: 11,
              color: "#ff4d6d",
            }}
            formatter={(v: any) => [Number(v).toFixed(6), "Huber Loss"]}
          />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#ff4d6d"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
