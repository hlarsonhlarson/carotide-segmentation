import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { request } from "./Segmentation";
export function Surface({
  base,
  side,
  onError,
  initialKind = "surface",
}: {
  initialKind?: string;
  base: string;
  side: string;
  onError: (s: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [version, setVersion] = useState(0),
    [kind, setKind] = useState(initialKind),
    [busy, setBusy] = useState(false),
    [status, setStatus] = useState("");
  const url =
    kind === "surface"
      ? `${base}/mesh/${side}`
      : `${base}/segment/${side}/su2_mesh/preview_stl`;
  useEffect(() => {
    if (!host.current || !version) return;
    const parent = host.current;
    let stopped = false;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#344152");
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    parent.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.DOLLY,
    };
    scene.add(new THREE.HemisphereLight(0xffffff, 0x36495c, 3));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(1, 2, 3);
    scene.add(light);
    let geometry: THREE.BufferGeometry | undefined;
    const material = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      roughness: 0.5,
      side: THREE.DoubleSide,
    });
    const abort = new AbortController();
    fetch("/api" + url, { signal: abort.signal })
      .then((r) => {
        if (!r.ok) throw Error("Сначала постройте геометрию");
        return r.arrayBuffer();
      })
      .then((data) => {
        if (stopped) return;
        geometry = new STLLoader().parse(data);
        geometry.center();
        geometry.computeBoundingSphere();
        const radius = geometry.boundingSphere?.radius || 1;
        scene.add(new THREE.Mesh(geometry, material));
        camera.position.set(radius * 2, radius, radius * 3);
        camera.near = radius / 1000;
        camera.far = radius * 100;
        camera.updateProjectionMatrix();
        controls.update();
      })
      .catch((e) => {
        if (!stopped) onError(String(e));
      });
    const resize = new ResizeObserver(() => {
      const w = parent.clientWidth;
      renderer.setSize(w, 420);
      camera.aspect = w / 420;
      camera.updateProjectionMatrix();
    });
    resize.observe(parent);
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });
    return () => {
      stopped = true;
      abort.abort();
      resize.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      geometry?.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url, version]);
  async function build(mesh: boolean) {
    setBusy(true);
    try {
      await request(
        `${base}/segment/${side}/${mesh ? "su2_mesh" : "build_surface"}`,
        {},
      );
      if (mesh) {
        let result;
        do {
          await new Promise((r) => setTimeout(r, 500));
          result = await request(`${base}/segment/${side}/su2_mesh/status`);
          setStatus(result.stage || result.state);
        } while (result.state === "running");
        if (result.state !== "done")
          throw Error(result.error || "Построение сетки прервано");
      }
      setKind(mesh ? "mesh" : "surface");
      setVersion((v) => v + 1);
      setStatus("Готово");
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section>
      <h2>3D-геометрия</h2>
      <label>
        Показывать
        <select
          value={kind}
          onChange={(e) => {
            setKind(e.target.value);
            setVersion(0);
          }}
        >
          <option value="surface">Поверхность</option>
          <option value="mesh">Расчётная сетка</option>
        </select>
      </label>
      <button disabled={busy} onClick={() => build(false)}>
        Построить поверхность
      </button>
      <button disabled={busy} onClick={() => build(true)}>
        Построить расчётную сетку
      </button>
      <button disabled={busy} onClick={() => setVersion((v) => v + 1)}>
        Показать сохранённую геометрию
      </button>
      <p>{status} · Вращение мышью, масштаб колесом</p>
      <div ref={host} />
      {version > 0 && (
        <a href={"/api" + url} download={side + ".stl"}>
          Скачать STL
        </a>
      )}
      {kind === "mesh" && (
        <p>
          <a href={"/api" + base + `/segment/${side}/su2_mesh/vtu`} download>
            Скачать VTU
          </a>{" "}
          ·{" "}
          <a href={"/api" + base + `/segment/${side}/su2_mesh/vtp`} download>
            Скачать VTP
          </a>
        </p>
      )}
    </section>
  );
}
