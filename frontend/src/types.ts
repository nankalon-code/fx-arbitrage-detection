// WebSocket message types
export interface Metrics {
  tick_count: number;
  arb_detected: number;
  total_pnl: number;
  trades: number;
  win_rate: number;
  epsilon: number;
  uptime?: number;
  throughput?: number;
  fill_rate?: number;
  n_pairs?: number;
}

export interface ArbEntry {
  tick: number;
  profit_bps: number;
  path: string[];
  legs: number;
}

export interface Opportunity {
  path: string[];
  profit_bps: number;
  profit_pct: number;
  legs: number;
}

export interface AgentStats {
  steps: number;
  epsilon: number;
  mean_reward_500: number;
  total_reward: number;
  mean_loss_100: number;
  buffer_size: number;
  sharpe_500: number;
}

export interface PriceData {
  bid: number;
  ask: number;
  mid: number;
  vol: number;
}

export interface TickFrame {
  type: "tick";
  tick: number;
  running: boolean;
  metrics: Metrics;
  opportunities: Opportunity[];
  arb_log: ArbEntry[];
  pnl_history: number[];
  loss_history: number[];
  price_chart: Record<string, number[]>;
  action: string;
  reward: number;
  position: boolean;
  agent_stats: AgentStats;
  prices: Record<string, PriceData>;
}

export interface InitFrame {
  type: "init";
  running: boolean;
  config: EngineConfig;
  fx_pairs: string[];
}

export interface StopFrame {
  type: "stopped";
}

export type WsFrame = TickFrame | InitFrame | StopFrame;

export interface EngineConfig {
  n_pairs: number;
  min_profit_bps: number;
  arb_prob: number;
  speed: number;
}

export type ConnectionStatus = "connected" | "connecting" | "disconnected" | "error";
