import { useRef, useMemo, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Float } from "@react-three/drei";
import * as THREE from "three";
import type { Opportunity } from "../types";

// ── Currency node positions on a sphere ─────────────────────────────────────
const CURRENCIES = [
  { id: "EUR", lat: 48.8, lon: 2.3, color: "#4fc3f7" },
  { id: "USD", lat: 38.9, lon: -77.0, color: "#00ff88" },
  { id: "JPY", lat: 35.7, lon: 139.7, color: "#ff4d6d" },
  { id: "GBP", lat: 51.5, lon: -0.1, color: "#a78bfa" },
  { id: "CHF", lat: 46.9, lon: 7.4, color: "#ffb300" },
  { id: "AUD", lat: -33.9, lon: 151.2, color: "#22d3ee" },
  { id: "CAD", lat: 45.4, lon: -75.7, color: "#f472b6" },
  { id: "NZD", lat: -41.3, lon: 174.8, color: "#34d399" },
];

function latLonToVec3(lat: number, lon: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

// ── Animated arc between two currency nodes ─────────────────────────────────
function ArcLine({
  start,
  end,
  isArb,
  intensity,
}: {
  start: THREE.Vector3;
  end: THREE.Vector3;
  isArb: boolean;
  intensity: number;
}) {
  const lineObj = useMemo(() => {
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    mid.normalize().multiplyScalar(start.length() * 1.3);
    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    const points = curve.getPoints(32);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: isArb ? "#00ff88" : "#1a2a4a",
      transparent: true,
      opacity: isArb ? 0.8 : 0.15,
    });
    return new THREE.Line(geometry, material);
  }, [start, end, isArb]);

  useFrame(({ clock }) => {
    if (isArb && lineObj.material instanceof THREE.LineBasicMaterial) {
      const t = clock.getElapsedTime();
      lineObj.material.opacity = 0.4 + Math.sin(t * 4 + intensity) * 0.3;
    }
  });

  return <primitive object={lineObj} />;
}

// ── Glowing particle traveling along arb path ───────────────────────────────
function TravelingParticle({
  curve,
  speed,
  color,
}: {
  curve: THREE.QuadraticBezierCurve3;
  speed: number;
  color: string;
}) {
  const ref = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = (clock.getElapsedTime() * speed) % 1;
    const point = curve.getPoint(t);
    ref.current.position.copy(point);
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.03, 8, 8]} />
      <meshBasicMaterial color={color} transparent opacity={0.9} />
    </mesh>
  );
}

// ── Currency Node ───────────────────────────────────────────────────────────
function CurrencyNode({
  position,
  label,
  color,
  isArb,
  scale,
}: {
  position: THREE.Vector3;
  label: string;
  color: string;
  isArb: boolean;
  scale: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (meshRef.current && isArb) {
      const t = clock.getElapsedTime();
      meshRef.current.scale.setScalar(scale * (1 + Math.sin(t * 3) * 0.15));
    }
    if (glowRef.current && isArb) {
      const t = clock.getElapsedTime();
      (glowRef.current.material as THREE.MeshBasicMaterial).opacity =
        0.15 + Math.sin(t * 2) * 0.1;
    }
  });

  return (
    <group position={position}>
      {/* Outer glow ring */}
      {isArb && (
        <mesh ref={glowRef}>
          <sphereGeometry args={[0.18, 16, 16]} />
          <meshBasicMaterial color="#00ff88" transparent opacity={0.15} />
        </mesh>
      )}
      {/* Core sphere */}
      <mesh ref={meshRef} scale={scale}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial
          color={isArb ? "#00ff88" : color}
          emissive={isArb ? "#00ff88" : color}
          emissiveIntensity={isArb ? 1.5 : 0.3}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>
      {/* Label */}
      <Float speed={1} floatIntensity={0.3}>
        <Text
          position={[0, 0.18, 0]}
          fontSize={0.07}
          color={isArb ? "#00ff88" : "#94a3b8"}
          anchorX="center"
          anchorY="bottom"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4xD-IQ-PuZJJXxfpAO-Lf1OQk6OThxPA.woff2"
          fontWeight={isArb ? 700 : 500}
          outlineWidth={0.005}
          outlineColor="#04070f"
        >
          {label}
        </Text>
      </Float>
    </group>
  );
}

// ── Wireframe Globe ─────────────────────────────────────────────────────────
function WireframeGlobe() {
  const ref = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y += 0.0008;
    }
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[2.0, 36, 24]} />
      <meshBasicMaterial
        color="#0d1d38"
        wireframe
        transparent
        opacity={0.08}
      />
    </mesh>
  );
}

// ── Pulse Ring at equator ───────────────────────────────────────────────────
function PulseRing() {
  const ref = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    const scale = 1 + Math.sin(t * 0.5) * 0.02;
    ref.current.scale.set(scale, scale, 1);
    (ref.current.material as THREE.MeshBasicMaterial).opacity =
      0.03 + Math.sin(t) * 0.015;
  });

  return (
    <mesh ref={ref} rotation={[Math.PI / 2, 0, 0]}>
      <ringGeometry args={[2.05, 2.08, 64]} />
      <meshBasicMaterial
        color="#4fc3f7"
        transparent
        opacity={0.04}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// ── Ambient Particles ───────────────────────────────────────────────────────
function AmbientParticles({ count = 200 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null);

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 3 + Math.random() * 2;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      arr[i * 3 + 2] = r * Math.cos(phi);
    }
    return arr;
  }, [count]);

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = clock.getElapsedTime() * 0.02;
      ref.current.rotation.x = Math.sin(clock.getElapsedTime() * 0.01) * 0.1;
    }
  });

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [positions]);

  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial
        color="#4fc3f7"
        size={0.015}
        transparent
        opacity={0.4}
        sizeAttenuation
      />
    </points>
  );
}

// ── Profit HUD overlay ──────────────────────────────────────────────────────
function ProfitHUD({ text }: { text: string }) {
  if (!text) return null;
  return (
    <Text
      position={[0, 2.6, 0]}
      fontSize={0.1}
      color="#00ff88"
      anchorX="center"
      font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4xD-IQ-PuZJJXxfpAO-Lf1OQk6OThxPA.woff2"
      fontWeight={700}
      outlineWidth={0.008}
      outlineColor="#04070f"
    >
      {text}
    </Text>
  );
}

// ── Main Scene ──────────────────────────────────────────────────────────────
function GlobeScene({
  opportunities,
}: {
  opportunities: Opportunity[];
}) {
  const RADIUS = 2.2;

  const nodePositions = useMemo(
    () =>
      CURRENCIES.map((c) => ({
        ...c,
        pos: latLonToVec3(c.lat, c.lon, RADIUS),
      })),
    []
  );

  // All edges between currencies
  const edges = useMemo(() => {
    const result: { from: string; to: string; start: THREE.Vector3; end: THREE.Vector3 }[] = [];
    for (let i = 0; i < nodePositions.length; i++) {
      for (let j = i + 1; j < nodePositions.length; j++) {
        result.push({
          from: nodePositions[i].id,
          to: nodePositions[j].id,
          start: nodePositions[i].pos,
          end: nodePositions[j].pos,
        });
      }
    }
    return result;
  }, [nodePositions]);

  // Determine arb currencies and edges
  const { arbCurrencies, arbEdgeSet, labelText, arbPaths } = useMemo(() => {
    const arbCurrencies = new Set<string>();
    const arbEdgeSet = new Set<string>();
    const arbPaths: { curve: THREE.QuadraticBezierCurve3; speed: number }[] = [];
    let labelText = "";

    if (opportunities.length > 0) {
      const best = opportunities[0];
      if (Array.isArray(best.path)) {
        best.path.forEach((c) => arbCurrencies.add(c));
        for (let i = 0; i < best.path.length - 1; i++) {
          const a = best.path[i];
          const b = best.path[i + 1];
          arbEdgeSet.add(`${a}-${b}`);
          arbEdgeSet.add(`${b}-${a}`);

          const startNode = nodePositions.find((n) => n.id === a);
          const endNode = nodePositions.find((n) => n.id === b);
          if (startNode && endNode) {
            const mid = new THREE.Vector3()
              .addVectors(startNode.pos, endNode.pos)
              .multiplyScalar(0.5);
            mid.normalize().multiplyScalar(RADIUS * 1.3);
            arbPaths.push({
              curve: new THREE.QuadraticBezierCurve3(startNode.pos, mid, endNode.pos),
              speed: 0.3 + i * 0.15,
            });
          }
        }
        labelText = `+${best.profit_bps.toFixed(1)} bps  |  ${best.path.join(" -> ")}`;
      }
    }
    return { arbCurrencies, arbEdgeSet, labelText, arbPaths };
  }, [opportunities, nodePositions]);

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={0.8} color="#4fc3f7" />
      <pointLight position={[-5, -3, 3]} intensity={0.4} color="#00ff88" />

      <WireframeGlobe />
      <PulseRing />
      <AmbientParticles />

      {/* Edges */}
      {edges.map((e) => {
        const key = `${e.from}-${e.to}`;
        const isArb = arbEdgeSet.has(key);
        return (
          <ArcLine
            key={key}
            start={e.start}
            end={e.end}
            isArb={isArb}
            intensity={Math.random() * Math.PI}
          />
        );
      })}

      {/* Traveling particles on arb paths */}
      {arbPaths.map((p, i) => (
        <TravelingParticle
          key={`particle-${i}`}
          curve={p.curve}
          speed={p.speed}
          color="#00ff88"
        />
      ))}

      {/* Currency nodes */}
      {nodePositions.map((c) => (
        <CurrencyNode
          key={c.id}
          position={c.pos}
          label={c.id}
          color={c.color}
          isArb={arbCurrencies.has(c.id)}
          scale={1}
        />
      ))}

      {/* Profit HUD */}
      <ProfitHUD text={labelText} />

      <OrbitControls
        enableZoom={true}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.4}
        minDistance={3}
        maxDistance={7}
        maxPolarAngle={Math.PI * 0.85}
        minPolarAngle={Math.PI * 0.15}
      />
    </>
  );
}

// ── Performance Stats Overlay ───────────────────────────────────────────────
function StatsOverlay({
  opportunities,
  tickCount,
}: {
  opportunities: Opportunity[];
  tickCount: number;
}) {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 12,
        left: 12,
        right: 12,
        display: "flex",
        justifyContent: "space-between",
        pointerEvents: "none",
        zIndex: 2,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--text-muted)",
          background: "rgba(4,7,15,0.7)",
          padding: "4px 10px",
          borderRadius: 6,
          border: "1px solid var(--bg-border-dim)",
        }}
      >
        {opportunities.length > 0 ? (
          <span style={{ color: "var(--accent-green)" }}>
            {opportunities.length} cycle{opportunities.length !== 1 ? "s" : ""} detected
          </span>
        ) : (
          <span>Scanning {CURRENCIES.length} nodes…</span>
        )}
      </div>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--text-muted)",
          background: "rgba(4,7,15,0.7)",
          padding: "4px 10px",
          borderRadius: 6,
          border: "1px solid var(--bg-border-dim)",
        }}
      >
        TICK #{tickCount.toLocaleString()}
      </div>
    </div>
  );
}

// ── Exported Component ──────────────────────────────────────────────────────
interface Props {
  opportunities: Opportunity[];
  prices: Record<string, { bid: number; ask: number; mid: number; vol: number }>;
  tickCount?: number;
}

export function GlobeNetwork({ opportunities, prices, tickCount = 0 }: Props) {
  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: 420,
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
        background: "radial-gradient(ellipse at center, #060d1e 0%, #04070f 70%)",
      }}
    >
      <Canvas
        camera={{ position: [0, 1.5, 4.5], fov: 50, near: 0.1, far: 100 }}
        style={{ background: "transparent" }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
      >
        <GlobeScene opportunities={opportunities} />
      </Canvas>

      <StatsOverlay opportunities={opportunities} tickCount={tickCount} />

      {/* Corner accent */}
      <div
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          color: "var(--accent-blue)",
          opacity: 0.5,
          letterSpacing: "1px",
          textTransform: "uppercase",
          pointerEvents: "none",
        }}
      >
        3D Currency Network
      </div>
    </div>
  );
}
