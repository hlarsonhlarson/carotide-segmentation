import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  labelVesselSegments,
  scenePoint,
  sectionPixelToMm,
  vesselSegmentNames,
  vesselSegments,
  wallGeometry,
  wallRings,
  type DisplayFlips,
  type SegmentationPoint,
  type VesselSegment,
} from "./segmentationGeometry";
export const branches = ["cca_ica", "cca_eca"];
export const branchColors: Record<string, string> = {
  cca_ica: "#f21c29",
  cca_eca: "#00ffff",
};
export const segmentColors: Record<VesselSegment, string> = {
  cca: "#ffa726",
  bifurcation: "#ce93d8",
  ica: branchColors.cca_ica,
  eca: branchColors.cca_eca,
};
type PlaqueKind = "calcified" | "lipid";
const plaqueKinds: PlaqueKind[] = ["calcified", "lipid"];
// White calcium stands out against the red ICA wall.
const plaqueColors: Record<PlaqueKind, string> = { calcified: "#ffffff", lipid: "#ffd54f" };
const plaqueNames: Record<PlaqueKind, string> = { calcified: "Кальций", lipid: "Атерома" };
const zero = [0, 0, 0];
type Runtime = {
  group: THREE.Group;
  fit: () => void;
  render: () => void;
  updateClipping: () => void;
  pointCount: number;
  contourCount: number;
  visibleKey: string;
  framed: boolean;
};
function clear(group: THREE.Group) {
  group.traverse((obj) => {
    if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.Points) {
      obj.geometry.dispose();
      const materials = Array.isArray(obj.material)
        ? obj.material
        : [obj.material];
      materials.forEach((material) => material.dispose());
    }
  });
  group.clear();
}
export function PathPreview({
  paths,
  points,
  spacing,
  side,
  displayFlips,
  origin = zero,
  segmentations,
}: {
  paths: Record<string, number[][]>;
  points: Record<string, number[][]>;
  spacing: number[];
  side: string;
  displayFlips?: DisplayFlips;
  origin?: number[];
  segmentations: Record<string, SegmentationPoint[]>;
}) {
  const host = useRef<HTMLDivElement>(null);
  const runtime = useRef<Runtime | null>(null);
  const [error, setError] = useState("");
  const [visible, setVisible] = useState<Record<VesselSegment, boolean>>(
    { cca: true, bifurcation: true, ica: true, eca: true });
  const [bifurcationMm, setBifurcationMm] = useState(10);
  const [plaqueVisible, setPlaqueVisible] = useState<Record<PlaqueKind, boolean>>(
    { calcified: true, lipid: true });
  // O(sections²) distance checks: recompute only when the series change,
  // not on every render.
  const labels = useMemo(
    () => labelVesselSegments(
      segmentations.cca_ica || [], segmentations.cca_eca || [], bifurcationMm),
    [segmentations, bifurcationMm],
  );
  const visibleKey = vesselSegments.filter((segment) => visible[segment]).join(",");
  // Sections whose plaque composition can be placed in 3D (needs the frame
  // stored with wall analyses since this version).
  const plaqueSections = branches.reduce((n, b) =>
    n + (segmentations[b] || []).filter((p, i) =>
      p.wall_analysis?.composition && p.wall_analysis.section_frame &&
      i >= (b === "cca_eca" ? labels.shared.cca_eca : 0)).length, 0);
  const sectionCounts = vesselSegments.map(segment =>
    branches.reduce((n, b) => n + (labels[b as "cca_ica" | "cca_eca"] || [])
      .filter((label, i) => label === segment &&
        i >= (b === "cca_eca" ? labels.shared.cca_eca : 0)).length, 0));
  const count = branches.reduce(
    (n, b) => n + (points[side + ":" + b]?.length || 0),
    0,
  );
  const contourCount = branches.reduce(
    (n, b) =>
      n +
      (segmentations[b] || []).filter((p) => (p.contour_mm?.length || 0) >= 3)
        .length,
    0,
  );
  const wallCounts = branches.map((b) =>
    (segmentations[b] || []).reduce(
      (n, p, i, series) =>
        n +
        Number(
          i > 0 &&
            (p.contour_mm?.length || 0) >= 3 &&
            (series[i - 1].contour_mm?.length || 0) >= 3,
        ),
      0,
    ),
  );

  useEffect(() => {
    const parent = host.current;
    if (!parent) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setError("3D недоступно: включите WebGL в браузере.");
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.domElement.setAttribute(
      "aria-label",
      "3D-путь и сегментированная стенка",
    );
    parent.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#344152");
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.DOLLY,
    };
    scene.add(new THREE.HemisphereLight(0xffffff, 0x344152, 3));
    const light = new THREE.DirectionalLight(0xffffff, 2);
    light.position.set(1, 2, 3);
    scene.add(light);
    const group = new THREE.Group();
    scene.add(group);
    // Frame the visible walls when there are any; otherwise the whole path.
    const subject = () => {
      const box = new THREE.Box3();
      group.children.forEach((child) => {
        if (child.userData.wall) box.expandByObject(child);
      });
      return box.isEmpty() ? new THREE.Box3().setFromObject(group) : box;
    };
    const bounds = () => {
      const box = subject();
      return box.isEmpty()
        ? new THREE.Sphere(new THREE.Vector3(), 10)
        : box.expandByScalar(1).getBoundingSphere(new THREE.Sphere());
    };
    const render = () => renderer.render(scene, camera);
    const updateClipping = () => {
      const sphere = bounds();
      camera.near = Math.max(0.001, sphere.radius / 10000);
      camera.far = Math.max(
        1000,
        sphere.radius * 20 + camera.position.distanceTo(sphere.center),
      );
      camera.updateProjectionMatrix();
    };
    const fit = () => {
      const sphere = bounds();
      const box = subject();
      const direction = new THREE.Vector3(0.35, 0.1, 1).normalize();
      const right = new THREE.Vector3(0, 1, 0).cross(direction).normalize();
      const up = direction.clone().cross(right);
      const tanVertical = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const tanHorizontal = tanVertical * camera.aspect;
      let distance = 10;
      if (!box.isEmpty()) {
        // Fit the actual tall, narrow vessel, rather than an enclosing sphere
        // whose width makes the model unnecessarily tiny in this narrow pane.
        box.expandByScalar(1);
        for (const x of [box.min.x, box.max.x])
          for (const y of [box.min.y, box.max.y])
            for (const z of [box.min.z, box.max.z]) {
              const offset = new THREE.Vector3(x, y, z).sub(sphere.center);
              distance = Math.max(
                distance,
                Math.abs(offset.dot(right)) / tanHorizontal +
                  offset.dot(direction),
                Math.abs(offset.dot(up)) / tanVertical + offset.dot(direction),
              );
            }
      }
      controls.target.copy(sphere.center);
      camera.position
        .copy(sphere.center)
        .addScaledVector(direction, distance * 1.1);
      camera.up.set(0, 1, 0);
      updateClipping();
      controls.update();
      render();
    };
    runtime.current = {
      group,
      fit,
      render,
      updateClipping,
      pointCount: 0,
      contourCount: 0,
      visibleKey: "",
      framed: false,
    };
    const resize = () => {
      const w = Math.max(parent.clientWidth, 1),
        h = Math.max(parent.clientHeight, 1);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      render();
    };
    controls.addEventListener("change", render);
    const observer = new ResizeObserver(resize);
    observer.observe(parent);
    resize();
    fit();
    return () => {
      runtime.current = null;
      observer.disconnect();
      controls.dispose();
      clear(group);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    const view = runtime.current;
    if (!view) return;
    clear(view.group);
    // Path samples use voxel indices; wall contours use world-mm. Both must
    // land in the same origin-relative scene, including reversed acquisitions.
    const voxel = (p: number[]) =>
      scenePoint(
        p.map((v, i) => v * spacing[i]),
        zero,
        displayFlips,
      );
    const world = (p: number[]) => scenePoint(p, origin, displayFlips);
    for (const branch of branches) {
      const key = side + ":" + branch;
      const path = paths[key] || points[key] || [];
      if (path.length > 1)
        view.group.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(path.map(voxel)),
            new THREE.LineBasicMaterial({ color: branchColors[branch] }),
          ),
        );
      for (const p of points[key] || []) {
        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(Math.max(...spacing) * 0.65, 12, 8),
          new THREE.MeshBasicMaterial({ color: branchColors[branch] }),
        );
        dot.position.copy(voxel(p));
        view.group.add(dot);
      }
      const series = segmentations[branch] || [];
      const branchLabels = labels[branch as "cca_ica" | "cca_eca"];
      // The common CCA is in both series; draw it once, from the ICA series.
      const skip = branch === "cca_eca" ? labels.shared.cca_eca : 0;
      const rings = wallRings(series, world);
      for (const segment of vesselSegments) {
        if (!visible[segment]) continue;
        // Interval i (sections i..i+1) takes the label of its distal section.
        const wall = wallGeometry(series, world, (i) =>
          i + 1 >= skip && branchLabels[i + 1] === segment, rings);
        if (wall.segments)
          view.group.add(
            Object.assign(new THREE.Mesh(
              wall.geometry,
              new THREE.MeshStandardMaterial({
                color: segmentColors[segment],
                transparent: true,
                opacity: 0.55,
                depthWrite: false,
                side: THREE.DoubleSide,
                roughness: 0.65,
              }),
            ), { userData: { wall: true } }),
          );
        else wall.geometry.dispose();
      }
      // Plaque components as points at their pixel positions in each section.
      const clouds: Record<PlaqueKind, number[]> = { calcified: [], lipid: [] };
      series.forEach((section, i) => {
        const wallResult = section.wall_analysis;
        const frame = wallResult?.section_frame, composition = wallResult?.composition;
        if (i < skip || !frame || !composition || !visible[branchLabels[i]]) return;
        for (const kind of plaqueKinds) {
          if (!plaqueVisible[kind]) continue;
          const pixels = kind === "calcified" ? composition.calcified_pixels : composition.lipid_pixels;
          for (const pixel of pixels) {
            const q = world(sectionPixelToMm(pixel, frame));
            clouds[kind].push(q.x, q.y, q.z);
          }
        }
      });
      for (const kind of plaqueKinds) {
        if (!clouds[kind].length) continue;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(clouds[kind], 3));
        view.group.add(new THREE.Points(geometry, new THREE.PointsMaterial({
          color: plaqueColors[kind], size: 0.45, sizeAttenuation: true,
          // Plaque lies inside the translucent wall; draw it on top.
          depthTest: false,
        })));
      }
    }
    view.updateClipping();
    if (
      count &&
      (!view.framed ||
        view.pointCount !== count ||
        view.contourCount !== contourCount ||
        view.visibleKey !== visibleKey)
    ) {
      view.fit();
      view.framed = true;
    }
    view.pointCount = count;
    view.contourCount = contourCount;
    view.visibleKey = visibleKey;
    view.render();
  }, [
    paths,
    points,
    spacing,
    side,
    displayFlips,
    origin,
    segmentations,
    count,
    contourCount,
    visible,
    bifurcationMm,
    plaqueVisible,
    labels,
  ]);

  return (
    <section className="preview-pane" aria-label="Путь и маска сегментации">
      <div className="pane-title">
        3D PATH <span>{count} точек</span>
        <button
          title="Вписать путь и стенку"
          aria-label="Вписать путь и стенку"
          onClick={() => runtime.current?.fit()}
        >
          ⤢
        </button>
      </div>
      <div ref={host} className="path-canvas" />
      {error && <p role="alert">{error}</p>}
      {!count && (
        <div className="preview-hint">Добавьте точки пути на срезах</div>
      )}
      <div
        className="path-wall-status"
        role="status"
        title={branches
          .map((branch, i) => `${branch === "cca_ica" ? "ICA" : "ECA"}: ${wallCounts[i]} участков стенки`)
          .join("\n") + "\nЛевая кнопка — вращение, правая — масштаб, средняя — перемещение"}
      >
        {vesselSegments.map((segment, i) => (
          <button
            key={segment}
            type="button"
            className="segment-chip"
            aria-pressed={visible[segment]}
            title={`${visible[segment] ? "Скрыть" : "Показать"}: ${vesselSegmentNames[segment]}, ${sectionCounts[i]} сечений`}
            onClick={() => setVisible({ ...visible, [segment]: !visible[segment] })}
          >
            <i style={{ background: segmentColors[segment] }} />
            {vesselSegmentNames[segment] === "Бифуркация" ? "Бифурк." : vesselSegmentNames[segment]}
            <b>{sectionCounts[i]}</b>
          </button>
        ))}
        {plaqueSections > 0 && plaqueKinds.map((kind) => (
          <button
            key={kind}
            type="button"
            className="segment-chip"
            aria-pressed={plaqueVisible[kind]}
            title={`${plaqueVisible[kind] ? "Скрыть" : "Показать"}: ${plaqueNames[kind]} (по ${plaqueSections} сечениям со стенкой)`}
            onClick={() => setPlaqueVisible({ ...plaqueVisible, [kind]: !plaqueVisible[kind] })}
          >
            <i className="plaque-dot" style={{ background: plaqueColors[kind] }} />
            {plaqueNames[kind]}
          </button>
        ))}
        {labels.split ? (
          <label className="segment-zone" title="Зона бифуркации вдоль пути от развилки">
            ±
            <input type="number" min={0} max={30} step={1} value={bifurcationMm}
              aria-label="Зона бифуркации, мм"
              onChange={(e) => setBifurcationMm(Math.max(0, +e.target.value || 0))} />
            мм
          </label>
        ) : (
          <span className="segment-zone">Развилка не найдена</span>
        )}
      </div>
    </section>
  );
}
