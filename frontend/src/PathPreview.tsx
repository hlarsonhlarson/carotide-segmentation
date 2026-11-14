import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  scenePoint,
  wallGeometry,
  type DisplayFlips,
  type SegmentationPoint,
} from "./segmentationGeometry";
export const branches = ["cca_ica", "cca_eca"];
export const branchColors: Record<string, string> = {
  cca_ica: "#f21c29",
  cca_eca: "#00ffff",
};
const zero = [0, 0, 0];
type Runtime = {
  group: THREE.Group;
  fit: () => void;
  render: () => void;
  updateClipping: () => void;
  pointCount: number;
  contourCount: number;
  framed: boolean;
};
function clear(group: THREE.Group) {
  group.traverse((obj) => {
    if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
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
    const bounds = () => {
      const box = new THREE.Box3().setFromObject(group);
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
      const box = new THREE.Box3().setFromObject(group);
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
      const wall = wallGeometry(segmentations[branch] || [], world);
      if (wall.segments)
        view.group.add(
          new THREE.Mesh(
            wall.geometry,
            new THREE.MeshStandardMaterial({
              color: branchColors[branch],
              transparent: true,
              opacity: 0.5,
              depthWrite: false,
              side: THREE.DoubleSide,
              roughness: 0.65,
            }),
          ),
        );
      else wall.geometry.dispose();
    }
    view.updateClipping();
    if (
      count &&
      (!view.framed ||
        view.pointCount !== count ||
        view.contourCount !== contourCount)
    ) {
      view.fit();
      view.framed = true;
    }
    view.pointCount = count;
    view.contourCount = contourCount;
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
      <div className="path-wall-status" role="status">
        {branches.map((branch, i) => (
          <span key={branch}>
            <i style={{ background: branchColors[branch] }} />
            {branch === "cca_ica" ? "ICA" : "ECA"}: {wallCounts[i]} участков
            стенки
          </span>
        ))}
      </div>
      <small className="preview-footer">Вращение · масштаб · перемещение</small>
    </section>
  );
}
