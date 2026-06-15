import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

function ParticleField({ count = 300 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null);

  const { positions, speeds, sizes } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const speeds = new Float32Array(count);
    const sizes = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
      speeds[i] = 0.005 + Math.random() * 0.015;
      sizes[i] = 0.02 + Math.random() * 0.04;
    }
    return { positions, speeds, sizes };
  }, [count]);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const posAttr = ref.current.geometry.getAttribute("position");
    const arr = posAttr.array as Float32Array;
    const t = clock.getElapsedTime();

    for (let i = 0; i < count; i++) {
      arr[i * 3 + 1] += speeds[i];
      arr[i * 3] += Math.sin(t * 0.5 + i) * 0.001;
      if (arr[i * 3 + 1] > 10) arr[i * 3 + 1] = -10;
    }
    posAttr.needsUpdate = true;
    ref.current.rotation.y = t * 0.01;
  });

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    return geo;
  }, [positions, sizes]);

  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial
        color="#4fc3f7"
        size={0.03}
        transparent
        opacity={0.25}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

function DataGrid() {
  const ref = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.x = Math.PI / 2;
      ref.current.position.y = -3;
      const mat = ref.current.material as THREE.MeshBasicMaterial;
      mat.opacity = 0.04 + Math.sin(clock.getElapsedTime() * 0.3) * 0.01;
    }
  });

  return (
    <mesh ref={ref}>
      <planeGeometry args={[30, 30, 30, 30]} />
      <meshBasicMaterial color="#4fc3f7" wireframe transparent opacity={0.04} />
    </mesh>
  );
}

function FloatingLines() {
  const ref = useRef<THREE.Group>(null);

  const lines = useMemo(() => {
    const result: { points: THREE.Vector3[]; color: string }[] = [];
    for (let i = 0; i < 8; i++) {
      const y = (Math.random() - 0.5) * 10;
      const z = -5 + Math.random() * 3;
      const start = new THREE.Vector3(-10, y, z);
      const end = new THREE.Vector3(10, y + (Math.random() - 0.5) * 2, z);
      const mid = new THREE.Vector3(0, y + (Math.random() - 0.5) * 3, z - 1);
      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      result.push({
        points: curve.getPoints(50),
        color: i % 2 === 0 ? "#00ff88" : "#4fc3f7",
      });
    }
    return result;
  }, []);

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.1) * 0.1;
    }
  });

  const lineObjects = useMemo(() => {
    return lines.map((l) => {
      const geo = new THREE.BufferGeometry();
      const arr = new Float32Array(l.points.flatMap((p) => [p.x, p.y, p.z]));
      geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
      const mat = new THREE.LineBasicMaterial({ color: l.color, transparent: true, opacity: 0.04 });
      return new THREE.Line(geo, mat);
    });
  }, [lines]);

  return (
    <group ref={ref}>
      {lineObjects.map((obj, i) => (
        <primitive key={i} object={obj} />
      ))}
    </group>
  );
}

export function ParticleBackground() {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 8], fov: 60, near: 0.1, far: 100 }}
        style={{ background: "transparent" }}
        dpr={[1, 1.5]}
        gl={{ antialias: false, alpha: true }}
      >
        <ParticleField />
        <DataGrid />
        <FloatingLines />
      </Canvas>
    </div>
  );
}
