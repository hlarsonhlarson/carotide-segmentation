import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
export function ResultMesh({
  points,
  triangles,
  values,
}: {
  points: number[][];
  triangles: number[][];
  values: number[];
}) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const parent = host.current;
    if (!parent) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#344152");
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(points.flat(), 3),
    );
    geometry.setIndex(triangles.flat());
    const min = values.reduce((a, b) => Math.min(a, b), Infinity),
      max = values.reduce((a, b) => Math.max(a, b), -Infinity);
    const colors = values.flatMap((v) => {
      const c = new THREE.Color().setHSL(
        (1 - (v - min) / (max - min || 1)) * 0.65,
        0.8,
        0.5,
      );
      return [c.r, c.g, c.b];
    });
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    geometry.center();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(geometry, material));
    const radius = geometry.boundingSphere?.radius || 1;
    const camera = new THREE.PerspectiveCamera(
      40,
      1,
      radius / 1000,
      radius * 100,
    );
    camera.position.set(radius * 2, radius, radius * 3);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    parent.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.DOLLY,
    };
    controls.update();
    const resize = new ResizeObserver(() => {
      renderer.setSize(parent.clientWidth, 420);
      camera.aspect = parent.clientWidth / 420;
      camera.updateProjectionMatrix();
    });
    resize.observe(parent);
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });
    return () => {
      resize.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [points, triangles, values]);
  return <div ref={host} aria-label="Карта результатов CFD" />;
}
