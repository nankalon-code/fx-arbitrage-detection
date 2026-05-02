import { useState, useEffect, useRef, useCallback } from "react";
import type { WsFrame, TickFrame, EngineConfig } from "../types";

const WS_URL = "ws://localhost:8000/ws";
const API_URL = "http://localhost:8000";

export function useWebSocket() {
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [lastFrame, setLastFrame] = useState<TickFrame | null>(null);
  const [config, setConfig] = useState<EngineConfig>({
    n_pairs: 10,
    min_profit_bps: 5,
    arb_prob: 0.07,
    speed: 10,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (event) => {
      try {
        const frame: WsFrame = JSON.parse(event.data);
        if (frame.type === "init") {
          setRunning(frame.running);
          setConfig(frame.config);
        } else if (frame.type === "tick") {
          setRunning(frame.running);
          setLastFrame(frame);
        } else if (frame.type === "stopped") {
          setRunning(false);
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 2s
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const startEngine = useCallback(async () => {
    await fetch(`${API_URL}/api/start`, { method: "POST" });
  }, []);

  const stopEngine = useCallback(async () => {
    await fetch(`${API_URL}/api/stop`, { method: "POST" });
  }, []);

  const updateConfig = useCallback(async (updates: Partial<EngineConfig>) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    await fetch(`${API_URL}/api/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newConfig),
    });
  }, [config]);

  return {
    connected,
    running,
    lastFrame,
    config,
    startEngine,
    stopEngine,
    updateConfig,
  };
}
