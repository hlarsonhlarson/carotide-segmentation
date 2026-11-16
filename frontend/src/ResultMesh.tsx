import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ParaView-style "Jet" colormap, as used in published carotid CFD figures.
const JET: [number, number, number, number][] = [
  [0.0, 0.0, 0.0, 0.56],
  [0.11, 0.0, 0.0, 1.0],
  [0.36, 0.0, 1.0, 1.0],
  [0.61, 1.0, 1.0, 0.0],
  [0.86, 1.0, 0.0, 0.0],
  [1.0, 0.5, 0.0, 0.0],
];
export function jet(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  for (let i = 1; i < JET.length; i++) {
    if (x <= JET[i][0]) {
      const [a, b] = [JET[i - 1], JET[i]];
      const f = (x - a[0]) / (b[0] - a[0] || 1);
      return [a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3])];
    }
  }
  return [0.5, 0, 0];
}
const gradientCss = `linear-gradient(to top, ${JET.map(([p, r, g, b]) =>
  `rgb(${r * 255},${g * 255},${b * 255}) ${p * 100}%`).join(", ")})`;

export type Streamlines = { points_mm: number[]; offsets: number[]; speed_cms: number[]; time_s: number };
type View = "anterior" | "posterior" | "right" | "left";
const VIEWS: Record<View, { dir: [number, number, number]; label: string }> = {
  // Patient LPS coordinates: +x left, +y posterior, +z superior.
  anterior: { dir: [0, -1, 0], label: "Спереди" },
  posterior: { dir: [0, 1, 0], label: "Сзади" },
  right: { dir: [-1, 0, 0], label: "Справа" },
  left: { dir: [1, 0, 0], label: "Слева" },
};

// Background presets; the colourbar text switches to light on dark ones.
export const BACKGROUNDS: { value: string; label: string }[] = [
  { value: "#ffffff", label: "Белый" },
  { value: "#d9dce1", label: "Светло-серый" },
  { value: "#52576b", label: "Серо-синий" },
  { value: "#000000", label: "Чёрный" },
];
const BG_KEY = "cfd-viewer-background";
function storedBackground() {
  try { return localStorage.getItem(BG_KEY) || "#ffffff"; } catch { return "#ffffff"; }
}
function isDark(hex: string) {
  const c = new THREE.Color(hex);
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b < 0.45;
}

function ticks(lo: number, hi: number, n = 6) {
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
}
function fmt(v: number, lo: number, hi: number) {
  const span = Math.abs(hi - lo);
  if (span >= 100) return v.toFixed(0);
  if (span >= 10) return v.toFixed(1);
  if (span >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

export function ResultMesh({
  points,
  triangles,
  values,
  range,
  title,
  streamlines,
  streamRange,
  fileName = "cfd-result",
  caption,
}: {
  points: number[][];
  triangles: number[][];
  values: ArrayLike<number>;
  range: [number, number];
  title: string;
  streamlines?: Streamlines | null;
  streamRange?: [number, number];
  fileName?: string;
  caption?: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{
    setView: (v: View) => void;
    snapshot: () => HTMLCanvasElement;
    colors: THREE.BufferAttribute;
    scene: THREE.Scene;
  } | null>(null);
  const [view, setView] = useState<View>("anterior");
  const [background, setBackground] = useState(storedBackground);
  const dark = isDark(background);
  const showLines = !!streamlines && !!streamRange;
  // With streamlines the colourbar describes velocity, as in the figures.
  const barRange = showLines ? streamRange! : range;
  const barTitle = showLines ? "Скорость, см/с" : title;

  useEffect(() => {
    const parent = host.current;
    if (!parent) return;
    const height = 520;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(background);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(points.flat(), 3));
    geometry.setIndex(triangles.flat());
    // Colours are written in place by the effect below, so playing frames
    // does not rebuild the scene.
    const colors = new THREE.Float32BufferAttribute(new Float32Array(points.length * 3).fill(0.82), 3);
    colors.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("color", colors);
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const box = geometry.boundingBox!;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.55,
      metalness: 0.0,
      transparent: showLines,
      opacity: showLines ? 0.22 : 1,
      depthWrite: !showLines,
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    const disposables: { dispose: () => void }[] = [geometry, material];
    if (showLines) {
      const s = streamlines!;
      const [vlo, vhi] = streamRange!;
      const positions: number[] = [], lineColors: number[] = [];
      for (let k = 0; k + 1 < s.offsets.length; k++) {
        for (let i = s.offsets[k]; i + 1 < s.offsets[k + 1]; i++) {
          for (const j of [i, i + 1]) {
            positions.push(s.points_mm[3 * j], s.points_mm[3 * j + 1], s.points_mm[3 * j + 2]);
            lineColors.push(...jet((s.speed_cms[j] - vlo) / (vhi - vlo || 1)));
          }
        }
      }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      lg.setAttribute("color", new THREE.Float32BufferAttribute(lineColors, 3));
      const lm = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
      scene.add(new THREE.LineSegments(lg, lm));
      disposables.push(lg, lm);
    }
    scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc4, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    scene.add(key);

    const camera = new THREE.PerspectiveCamera(22, 1, 0.1, 10000);
    camera.up.set(0, 0, 1); // superior up
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    parent.prepend(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.DOLLY };
    controls.target.copy(center);

    function place(v: View) {
      const dir = new THREE.Vector3(...VIEWS[v].dir);
      const fov = THREE.MathUtils.degToRad(camera.fov);
      const across = Math.max(Math.abs(size.x * dir.y) + Math.abs(size.y * dir.x), 1);
      // Fill ~90% of the height, leaving the right side for the colourbar.
      const depth = Math.abs(size.x * dir.x) + Math.abs(size.y * dir.y);
      const fitHeight = (size.z * 1.08) / 2 / Math.tan(fov / 2);
      const fitWidth = (across * 1.6) / 2 / Math.tan(fov / 2) / camera.aspect;
      const dist = Math.max(fitHeight, fitWidth) + depth / 2;
      camera.position.copy(center).addScaledVector(dir, dist);
      camera.near = dist / 100;
      camera.far = dist * 10;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
    }
    const resize = new ResizeObserver(() => {
      const w = Math.max(parent.clientWidth, 1);
      renderer.setSize(w, height);
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
    });
    resize.observe(parent);
    renderer.setSize(Math.max(parent.clientWidth, 1), height);
    camera.aspect = Math.max(parent.clientWidth, 1) / height;
    place(view);
    renderer.setAnimationLoop(() => {
      controls.update();
      key.position.copy(camera.position).add(new THREE.Vector3(0, 0, size.z));
      renderer.render(scene, camera);
    });
    api.current = {
      colors,
      scene,
      setView: place,
      snapshot: () => {
        renderer.render(scene, camera);
        return renderer.domElement;
      },
    };
    return () => {
      api.current = null;
      resize.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      disposables.forEach((d) => d.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    };
    // view, colours and background are applied in place without rebuilding the scene
  }, [points, triangles, streamlines, streamRange?.[0], streamRange?.[1]]);

  useEffect(() => {
    const colors = api.current?.colors;
    if (!colors) return;
    const out = colors.array as Float32Array;
    const [lo, hi] = range;
    const n = Math.min(values.length, out.length / 3);
    for (let i = 0; i < n; i++) {
      const [r, g, b] = showLines ? [0.82, 0.82, 0.84] : jet((values[i] - lo) / (hi - lo || 1));
      out[3 * i] = r; out[3 * i + 1] = g; out[3 * i + 2] = b;
    }
    colors.needsUpdate = true;
  }, [values, range[0], range[1], showLines, points, triangles, streamlines, streamRange?.[0], streamRange?.[1]]);

  useEffect(() => {
    if (api.current) api.current.scene.background = new THREE.Color(background);
    try { localStorage.setItem(BG_KEY, background); } catch { /* per-viewer convenience only */ }
  }, [background, points, triangles, streamlines]);

  function exportPng() {
    const gl = api.current?.snapshot();
    if (!gl) return;
    const scale = 2;
    const barW = 150 * scale;
    const out = document.createElement("canvas");
    out.width = gl.width + barW;
    out.height = gl.height;
    const ctx = out.getContext("2d")!;
    const ink = dark ? "#f2f2f2" : "#111";
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(gl, 0, 0);
    const label = showLines ? `Линии тока, t = ${streamlines!.time_s.toFixed(2)} с (пик систолы)` : caption;
    if (label) {
      ctx.fillStyle = ink;
      ctx.font = `${13 * scale}px Arial`;
      ctx.textBaseline = "top";
      ctx.fillText(label, 12 * scale, 10 * scale);
    }
    const [lo, hi] = barRange;
    const x = gl.width + 20 * scale, top = out.height * 0.18, h = out.height * 0.64, w = 18 * scale;
    const grad = ctx.createLinearGradient(0, top + h, 0, top);
    JET.forEach(([p, r, g, b]) => grad.addColorStop(p, `rgb(${r * 255},${g * 255},${b * 255})`));
    ctx.fillStyle = grad;
    ctx.fillRect(x, top, w, h);
    ctx.strokeStyle = dark ? "#ddd" : "#333";
    ctx.lineWidth = scale;
    ctx.strokeRect(x, top, w, h);
    ctx.fillStyle = ink;
    ctx.font = `${12 * scale}px Arial`;
    ctx.textBaseline = "middle";
    ticks(lo, hi).forEach((t) => {
      const y = top + h - ((t - lo) / (hi - lo || 1)) * h;
      ctx.fillRect(x + w, y - scale / 2, 5 * scale, scale);
      ctx.fillText(fmt(t, lo, hi), x + w + 8 * scale, y);
    });
    ctx.save();
    ctx.translate(x + w + 70 * scale, top + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.font = `bold ${13 * scale}px Arial`;
    ctx.fillText(barTitle, 0, 0);
    ctx.restore();
    const a = document.createElement("a");
    a.download = `${fileName}-${view}.png`;
    a.href = out.toDataURL("image/png");
    a.click();
  }

  const [lo, hi] = barRange;
  return (
    <div className="result-viewer">
      <div ref={host} className={"result-canvas" + (dark ? " dark" : "")} aria-label="Карта результатов CFD">
        <div className="result-colorbar" aria-label="Цветовая шкала">
          <div className="bar" style={{ background: gradientCss }} />
          <div className="labels">
            {ticks(lo, hi).reverse().map((t) => <span key={t}>{fmt(t, lo, hi)}</span>)}
          </div>
          <div className="title">{barTitle}</div>
        </div>
        {showLines
          ? <div className="result-caption">Линии тока, t = {streamlines!.time_s.toFixed(2)} с (пик систолы)</div>
          : caption && <div className="result-caption">{caption}</div>}
      </div>
      <div className="button-row result-views">
        {(Object.keys(VIEWS) as View[]).map((v) => (
          <button key={v} aria-pressed={view === v} onClick={() => { setView(v); api.current?.setView(v); }}>
            {VIEWS[v].label}
          </button>
        ))}
        <span className="result-background" role="group" aria-label="Цвет фона">
          Фон
          {BACKGROUNDS.map((b) => (
            <button key={b.value} className="swatch" title={b.label} aria-label={`Фон: ${b.label}`}
              aria-pressed={background === b.value} style={{ background: b.value }}
              onClick={() => setBackground(b.value)} />
          ))}
          <input type="color" value={background} title="Свой цвет фона" aria-label="Свой цвет фона"
            onChange={(e) => setBackground(e.target.value)} />
        </span>
        <button className="primary" onClick={exportPng}>Сохранить PNG</button>
      </div>
    </div>
  );
}
