import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
export const branches = ["cca_ica", "cca_eca"];
export const branchColors: Record<string, string> = {
  cca_ica: "#f21c29",
  cca_eca: "#00ffff",
};
export function PathPreview({
  paths,
  points,
  spacing,
  side,
  displayFlips,
}: {
  paths: Record<string, number[][]>;
  points: Record<string, number[][]>;
  spacing: number[];
  displayFlips?: Record<string, { x: boolean; y: boolean }>;
  side: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const parent = host.current;
    if (!parent) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#344152");
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10000);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      parent.textContent = "3D недоступно: включите WebGL в браузере.";
      return () => {
        parent.textContent = "";
      };
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    parent.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.DOLLY,
    };
    const group = new THREE.Group();
    scene.add(group);
    const vector = (p: number[]) =>
      new THREE.Vector3(
        p[2] * spacing[2] * (displayFlips?.axial?.x ? -1 : 1),
        p[0] * spacing[0] * (displayFlips?.coronal?.y ? -1 : 1),
        p[1] * spacing[1] * (displayFlips?.axial?.y ? -1 : 1),
      );
    branches.forEach((branch) => {
      const key = side + ":" + branch;
      const path = paths[key] || points[key] || [];
      if (path.length > 1)
        group.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(path.map(vector)),
            new THREE.LineBasicMaterial({ color: branchColors[branch] }),
          ),
        );
      (points[key] || []).forEach((p) => {
        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(Math.max(...spacing) * 0.65, 12, 8),
          new THREE.MeshBasicMaterial({ color: branchColors[branch] }),
        );
        dot.position.copy(vector(p));
        group.add(dot);
      });
    });
    const bounds = new THREE.Box3().setFromObject(group);
    const center = bounds.isEmpty()
      ? new THREE.Vector3()
      : bounds.getCenter(new THREE.Vector3());
    const radius = bounds.isEmpty()
      ? 30
      : Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 5);
    controls.target.copy(center);
    const render = () => renderer.render(scene, camera);
    const resize = () => {
      const w = Math.max(parent.clientWidth, 1),
        h = Math.max(parent.clientHeight, 1);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      const distance =
        (radius /
          Math.sin(
            Math.atan(
              Math.tan(THREE.MathUtils.degToRad(19)) *
                Math.min(camera.aspect, 1),
            ),
          )) *
        1.15;
      camera.position
        .copy(center)
        .add(new THREE.Vector3(distance * 0.35, distance * 0.1, distance));
      camera.far = distance * 10;
      camera.updateProjectionMatrix();
      controls.update();
      render();
    };
    controls.addEventListener("change", render);
    const observer = new ResizeObserver(resize);
    observer.observe(parent);
    resize();
    return () => {
      observer.disconnect();
      controls.dispose();
      group.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          obj.geometry.dispose();
          const materials = Array.isArray(obj.material)
            ? obj.material
            : [obj.material];
          materials.forEach((m) => m.dispose());
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [paths, points, spacing, side, displayFlips]);
  const count = branches.reduce(
    (sum, b) => sum + (points[side + ":" + b]?.length || 0),
    0,
  );
  return (
    <section className="preview-pane">
      <div className="pane-title">
        3D PATH <span>{count} точек</span>
      </div>
      <div ref={host} className="path-canvas" />
      {!count && (
        <div className="preview-hint">Добавьте точки пути на срезах</div>
      )}
      <small className="preview-footer">Вращение · масштаб · перемещение</small>
    </section>
  );
}
