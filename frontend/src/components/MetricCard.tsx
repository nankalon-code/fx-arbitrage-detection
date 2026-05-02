interface Props {
  label: string;
  value: string | number;
  sub?: string;
  colorClass?: "green" | "blue" | "amber" | "red" | "purple";
  positive?: boolean;
  negative?: boolean;
}

export function MetricCard({ label, value, sub, colorClass, positive, negative }: Props) {
  let cardClass = "metric-card";
  if (positive) cardClass += " positive";
  if (negative) cardClass += " negative";

  let valClass = "metric-value";
  if (colorClass) valClass += ` ${colorClass}`;

  return (
    <div className={cardClass}>
      <div className="metric-label">{label}</div>
      <div className={valClass}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}
