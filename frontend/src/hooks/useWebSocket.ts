import { useState, useEffect, useRef, useCallback } from "react";
import type { WsFrame, TickFrame, EngineConfig, ConnectionStatus } from "../types";

const isProd = import.meta.env.PROD;
const PROD_URL = "fx-arbitrage-detection-up.railway.app";

const WS_URL = isProd ? `wss://${PROD_URL}/ws` : "ws://localhost:8000/ws";
const API_URL = isProd ? `https://${PROD_URL}` : "http://localhost:8000";

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 15000]; // exponential backoff

export function useWebSocket() {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [running, setRunning] = useState(false);
  const [lastFrame, setLastFrame] = useState<TickFrame | null>(null);
  const [config, setConfig] = useState<EngineConfig>({
    n_pairs: 20,
    min_profit_bps: 3,
    arb_prob: 0.10,
    speed: 25,
  });
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempt = useRef(0);
  const mountedRef = useRef(true);
  const frameBufferRef = useRef<TickFrame | null>(null);
  const rafRef = useRef<number | null>(null);

  // Throttle state updates to animation frame rate
  const flushFrame = useCallback(() => {
    if (frameBufferRef.current && mountedRef.current) {
      setLastFrame(frameBufferRef.current);
      frameBufferRef.current = null;
    }
    rafRef.current = null;
  }, []);

  const scheduleFlush = useCallback(() => {
    if (rafRef.current === null) {
      rafRef.current = requestAnimationFrame(flushFrame);
    }
  }, [flushFrame]);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      wsRef.current.close();
    }

    setConnectionStatus("connecting");
    setError(null);

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionStatus("connected");
        setError(null);
        reconnectAttempt.current = 0;
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const frame: WsFrame = JSON.parse(event.data);
          if (frame.type === "init") {
            setRunning(frame.running);
            setConfig(frame.config);
          } else if (frame.type === "tick") {
            setRunning(frame.running);
            // Buffer frame and flush on next animation frame
            frameBufferRef.current = frame;
            scheduleFlush();
          } else if (frame.type === "stopped") {
            setRunning(false);
          }
        } catch {
          // ignore malformed frames
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnectionStatus("disconnected");
        // Exponential backoff reconnect
        const delay = RECONNECT_DELAYS[
          Math.min(reconnectAttempt.current, RECONNECT_DELAYS.length - 1)
        ];
        reconnectAttempt.current += 1;
        reconnectTimer.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setConnectionStatus("error");
        setError("Connection failed. Backend may be offline.");
        ws.close();
      };
    } catch (e) {
      setConnectionStatus("error");
      setError("Failed to create WebSocket connection.");
    }
  }, [scheduleFlush]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Ping keepalive
  useEffect(() => {
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 25000);
    return () => clearInterval(interval);
  }, []);

  const startEngine = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/start`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (e) {
      setError("Failed to start engine. Check backend connection.");
    }
  }, []);

  const stopEngine = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/stop`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (e) {
      setError("Failed to stop engine.");
    }
  }, []);

  const updateConfig = useCallback(async (updates: Partial<EngineConfig>) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    try {
      const res = await fetch(`${API_URL}/api/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newConfig),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (e) {
      setError("Failed to update config.");
    }
  }, [config]);

  const clearError = useCallback(() => setError(null), []);

  return {
    connected: connectionStatus === "connected",
    connectionStatus,
    running,
    lastFrame,
    config,
    error,
    startEngine,
    stopEngine,
    updateConfig,
    clearError,
  };
}
