import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { branchColors } from "./PathPreview";
import {
  scenePoint,
  selectedRing,
  wallGeometry,
  type DisplayFlips,
  type SegmentationPoint,
} from "./segmentationGeometry";

const zero = [0, 0, 0];
type Runtime = {
  path: THREE.Line;
  wall: THREE.Mesh;
  ring: THREE.LineLoop;
  render: () => void;
  fit: () => void;
  updateClipping: () => void;
  contourCount: number;
  hasFramed: boolean;
};
export function SegmentationPreview({
  points,
  index,
  radius,
  branch,
  origin = zero,
  displayFlips,
}: {
  points: SegmentationPoint[];
  index: number;
  radius: number;
  branch: string;
  origin?: number[];
  displayFlips?: DisplayFlips;
}) {
  const host = useRef<HTMLDivElement>(null);
  const runtime = useRef<Runtime | null>(null);
  const [error, setError] = useState("");
  const color = branchColors[branch];
  const accepted = points.filter(
    (p) => (p.contour_mm?.length || 0) >= 3,
  ).length;
  const segments = points.reduce(
    (n, p, i) =>
      n +
      Number(
        i > 0 &&
          (p.contour_mm?.length || 0) >= 3 &&
          (points[i - 1].contour_mm?.length || 0) >= 3,
      ),
    0,
  );

  useEffect(() => {
    const parent = host.current;
    if (!parent) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setError("Не удалось открыть 3D. Проверьте поддержку WebGL в браузере.");
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.domElement.setAttribute(
      "aria-label",
      "3D-предпросмотр стенки сосуда",
    );
    parent.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#080a0e");
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
    const path = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial(),
    );
    const wall = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshStandardMaterial({
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
        side: THREE.DoubleSide,
        roughness: 0.65,
      }),
    );
    const ring = new THREE.LineLoop(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }),
    );
    ring.renderOrder = 10;
    const content = new THREE.Group();
    content.add(path, wall, ring);
    scene.add(content);
    const render = () => renderer.render(scene, camera);
    const bounds = () => {
      const box = new THREE.Box3().setFromObject(content);
      if (box.isEmpty()) return new THREE.Sphere(new THREE.Vector3(), 10);
      return box.expandByScalar(3).getBoundingSphere(new THREE.Sphere());
    };
    const updateClipping = () => {
      const sphere = bounds();
      camera.near = Math.max(sphere.radius / 10000, 0.001);
      camera.far = Math.max(
        1000,
        sphere.radius * 20 + camera.position.distanceTo(sphere.center),
      );
      camera.updateProjectionMatrix();
    };
    const fit = () => {
      const sphere = bounds();
      const halfAngle = Math.atan(
        Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) *
          Math.min(camera.aspect, 1),
      );
      const distance = (sphere.radius / Math.sin(halfAngle)) * 1.1;
      controls.target.copy(sphere.center);
      camera.position
        .copy(sphere.center)
        .add(new THREE.Vector3(0, 0, distance));
      camera.up.set(0, 1, 0);
      updateClipping();
      controls.update();
      render();
    };
    runtime.current = {
      path,
      wall,
      ring,
      render,
      fit,
      updateClipping,
      contourCount: 0,
      hasFramed: false,
    };
    const resize = () => {
      const width = Math.max(parent.clientWidth, 1),
        height = Math.max(parent.clientHeight, 1);
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      render();
    };
    controls.addEventListener("change", render);
    const observer = new ResizeObserver(resize);
    observer.observe(parent);
    resize();
    return () => {
      runtime.current = null;
      observer.disconnect();
      controls.dispose();
      [path, wall, ring].forEach((item) => {
        item.geometry.dispose();
        (item.material as THREE.Material).dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  // A different slice or contrast value does not rebuild the wall or camera.
  useEffect(() => {
    const view = runtime.current;
    if (!view) return;
    const convert = (p: number[]) => scenePoint(p, origin, displayFlips);
    view.path.geometry.dispose();
    view.path.geometry = new THREE.BufferGeometry().setFromPoints(
      points.map((p) => convert(p.point_mm)),
    );
    view.path.visible = points.length > 1;
    const result = wallGeometry(points, convert);
    view.wall.geometry.dispose();
    view.wall.geometry = result.geometry;
    view.wall.visible = result.segments > 0;
    (view.path.material as THREE.LineBasicMaterial).color.set(color);
    (view.wall.material as THREE.MeshStandardMaterial).color.set(color);
    view.updateClipping();
    if (points.length && (!view.hasFramed || accepted !== view.contourCount)) {
      view.fit();
      view.hasFramed = true;
    }
    view.contourCount = accepted;
    view.render();
  }, [points, origin, displayFlips, color, accepted]);

  useEffect(() => {
    const view = runtime.current;
    if (!view) return;
    const vertices = selectedRing(points[index], radius, (p) =>
      scenePoint(p, origin, displayFlips),
    );
    view.ring.geometry.dispose();
    view.ring.geometry = new THREE.BufferGeometry().setFromPoints(vertices);
    view.ring.visible = vertices.length > 0;
    view.updateClipping();
    view.render();
  }, [points, index, radius, origin, displayFlips]);

  return (
    <section className="wall-preview" aria-label="Предпросмотр стенки">
      <div className="pane-title">
        <span>WALL PREVIEW</span>
        <button
          aria-label="Вписать 3D-модель"
          onClick={() => runtime.current?.fit()}
        >
          ⤢
        </button>
      </div>
      <div className="wall-preview-canvas" ref={host} />
      {error && <p role="alert">{error}</p>}
      {!points.length && (
        <p className="wall-preview-hint">Подготовьте сечения вдоль пути</p>
      )}
      <div className="wall-preview-caption" role="status">
        <span>
          Сечение {points.length ? index + 1 : 0} / {points.length} ·{" "}
          {points[index]?.contour_mm?.length
            ? "готовый контур"
            : "кольцо радиуса"}
        </span>
        <span>
          {accepted} контуров · {segments} участков стенки
        </span>
      </div>
      <small>
        Белое кольцо — текущее сечение. Левая кнопка — вращение, правая —
        масштаб, средняя — перемещение.
      </small>
    </section>
  );
}
