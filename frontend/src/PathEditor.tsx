import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { request } from "./Segmentation";
import {
  segmentationKey,
  type SegmentationPoint,
  type SegmentationChange,
} from "./segmentationGeometry";
import { useImageNavigation } from "./useImageNavigation";
import { PathPreview, branches, branchColors } from "./PathPreview";
export type Axis = "axial" | "coronal" | "sagittal";
export function voxelAt(
  axis: Axis,
  index: number,
  row: number,
  col: number,
  shape: number[],
) {
  return axis === "axial"
    ? [index, row, col]
    : axis === "coronal"
      ? [shape[0] - 1 - row, index, col]
      : [shape[0] - 1 - row, col, index];
}
const axes: Axis[] = ["axial", "coronal", "sagittal"];
const axisNames = {
  axial: "Аксиальная",
  coronal: "Корональная",
  sagittal: "Сагиттальная",
};
function SlicePane({
  axis,
  index,
  study,
  base,
  wl,
  ww,
  points,
  side,
  start,
  indices,
  onIndex,
  onPoint,
}: {
  axis: Axis;
  index: number;
  study: any;
  base: string;
  wl: number;
  ww: number;
  points: Record<string, number[][]>;
  side: string;
  start: number[] | null;
  indices: number[];
  onIndex: (n: number) => void;
  onPoint: (p: number[]) => void;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const content = useRef<SVGGElement>(null);
  const dimension = axes.indexOf(axis);
  const rows = axis === "axial" ? study.shape[1] : study.shape[0];
  const cols = axis === "sagittal" ? study.shape[1] : study.shape[2];
  const sx = study.spacing_mm[axis === "sagittal" ? 1 : 2],
    sy = study.spacing_mm[axis === "axial" ? 1 : 0];
  const flips = study.display_flips?.[axis];
  const navigation = useImageNavigation(
    cols * sx,
    rows * sy,
    flips?.x,
    flips?.y,
  );
  const { zoom } = navigation;
  useEffect(() => {
    const element = viewport.current;
    function wheel(e: WheelEvent) {
      e.preventDefault();
      if (e.deltaY)
        onIndex(
          Math.max(
            0,
            Math.min(study.shape[dimension] - 1, index + Math.sign(e.deltaY)),
          ),
        );
    }
    element?.addEventListener("wheel", wheel, { passive: false });
    return () => element?.removeEventListener("wheel", wheel);
  }, [index, dimension, onIndex, study]);
  function marker(p: number[], color: string, key: string) {
    if (Math.abs(p[dimension] - index) >= 1) return null;
    return (
      <circle
        key={key}
        cx={((axis === "sagittal" ? p[1] : p[2]) + 0.5) * sx}
        cy={((axis === "axial" ? p[1] : study.shape[0] - 1 - p[0]) + 0.5) * sy}
        r={Math.max((cols * sx) / 110, 1)}
        fill={color}
        stroke="#fff"
        strokeWidth={0.5}
      />
    );
  }
  return (
    <section className={`slice-pane ${axis}`} aria-label={axisNames[axis]}>
      <div className="pane-title">
        {axis.toUpperCase()}
        <span className="pane-tools">
          <button
            aria-label={`Уменьшить: ${axisNames[axis]}`}
            disabled={zoom <= 0.5}
            onClick={() => navigation.zoomBy(1 / 1.4)}
          >
            −
          </button>
          <button
            aria-label={`Увеличить: ${axisNames[axis]}`}
            disabled={zoom >= 10}
            onClick={() => navigation.zoomBy(1.4)}
          >
            +
          </button>
          <button
            aria-label={`Перевернуть: ${axisNames[axis]}`}
            title="Повернуть изображение на 180°"
            onClick={navigation.rotate}
          >
            ↻
          </button>
          <button
            aria-label={`Вписать: ${axisNames[axis]}`}
            title="Сбросить масштаб, сдвиг и поворот"
            onClick={navigation.reset}
          >
            ⤢
          </button>
          <span>
            {index + 1} / {study.shape[dimension]}
          </span>
        </span>
      </div>
      <div className="slice-viewport" ref={viewport}>
        <svg
          viewBox={`0 0 ${cols * sx} ${rows * sy}`}
          aria-label={`${axisNames[axis]}: разметка`}
          {...navigation.handlers}
          onClick={(e) => {
            if (e.button !== 0 || !navigation.canClick()) return;
            const matrix = content.current?.getScreenCTM();
            if (!matrix) return;
            const point = new DOMPoint(e.clientX, e.clientY).matrixTransform(
              matrix.inverse(),
            );
            const row = Math.floor(point.y / sy),
              col = Math.floor(point.x / sx);
            if (row >= 0 && row < rows && col >= 0 && col < cols)
              onPoint(voxelAt(axis, index, row, col, study.shape));
          }}
        >
          <g ref={content} transform={navigation.transform}>
            <image
              href={`/api${base}/slice?axis=${axis}&index=${index}&wl=${wl}&ww=${ww}`}
              width={cols * sx}
              height={rows * sy}
              preserveAspectRatio="none"
            />
            <line
              x1={((axis === "sagittal" ? indices[1] : indices[2]) + 0.5) * sx}
              x2={((axis === "sagittal" ? indices[1] : indices[2]) + 0.5) * sx}
              y1={0}
              y2={rows * sy}
              stroke="#eff2f7"
              strokeOpacity={0.35}
              strokeWidth={0.7 / zoom}
            />
            <line
              x1={0}
              x2={cols * sx}
              y1={
                ((axis === "axial"
                  ? indices[1]
                  : study.shape[0] - 1 - indices[0]) +
                  0.5) *
                sy
              }
              y2={
                ((axis === "axial"
                  ? indices[1]
                  : study.shape[0] - 1 - indices[0]) +
                  0.5) *
                sy
              }
              stroke="#eff2f7"
              strokeOpacity={0.35}
              strokeWidth={0.7 / zoom}
            />
            {branches.flatMap((b) =>
              (points[side + ":" + b] || []).map((p, i) =>
                marker(p, branchColors[b], b + i),
              ),
            )}
            {start && marker(start, "#ffffff", "trace-start")}
          </g>
        </svg>
      </div>
      <input
        aria-label={`Срез: ${axisNames[axis]}`}
        type="range"
        min={0}
        max={study.shape[dimension] - 1}
        value={index}
        onChange={(e) => onIndex(+e.target.value)}
      />
    </section>
  );
}
export function PathEditor({
  base,
  side,
  branch,
  study,
  onError,
  onSide,
  onBranch,
  onStep,
  workflow,
  segmentations,
  onSeriesChange,
}: {
  base: string;
  side: string;
  branch: string;
  study: any;
  onError: (s: string) => void;
  onSide: (s: string) => void;
  onBranch: (s: string) => void;
  onStep: (n: number) => void;
  workflow: ReactNode;
  segmentations: Record<string, SegmentationPoint[]>;
  onSeriesChange: SegmentationChange;
}) {
  const [indices, setIndices] = useState<number[]>(
    study.shape.map((n: number) => Math.floor(n / 2)),
  );
  const [points, setPoints] = useState<Record<string, number[][]>>({});
  const [paths, setPaths] = useState<Record<string, number[][]>>({});
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState("navigate");
  const [start, setStart] = useState<number[] | null>(null);
  const [moving, setMoving] = useState<{
    branch: string;
    index: number;
  } | null>(null);
  const [wl, setWl] = useState<number>(study.default_window.wl);
  const [ww, setWw] = useState<number>(study.default_window.ww);
  const lock = useRef(false);
  const route = `${base}/centerline/${side}/${branch}`;
  const generation = useRef(0);
  const wallSeries = useMemo(
    () =>
      Object.fromEntries(
        branches.map((b) => [
          b,
          segmentations[segmentationKey(base, side, b)] || [],
        ]),
      ),
    [segmentations, base, side],
  );
  async function readState() {
    return Promise.all([
      request(base + "/control_points"),
      request(base + "/centerline"),
      ...branches.map((b) => request(`${base}/segment/${side}/${b}`)),
    ]);
  }
  function acceptState([p, path, ...series]: any[]) {
    setPoints(p);
    setPaths(path);
    branches.forEach((b, i) =>
      onSeriesChange(base, side, b, series[i].points || []),
    );
  }
  useEffect(() => {
    let active = true;
    const current = ++generation.current;
    readState()
      .then((data) => {
        if (active && current === generation.current) acceptState(data);
      })
      .catch((e) => {
        if (active) onError(String(e));
      });
    return () => {
      active = false;
    };
  }, [base, side]);
  async function action(fn: () => Promise<unknown>) {
    if (lock.current) return;
    lock.current = true;
    ++generation.current;
    setBusy(true);
    onError("");
    try {
      await fn();
      // A path mutation invalidates its contours on the server. Clear stale
      // overlays immediately, then restore the current per-branch state.
      branches.forEach((b) => onSeriesChange(base, side, b, []));
      acceptState(await readState());
    } catch (e) {
      onError(String(e));
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }
  function click(point: number[]) {
    if (busy) return;
    if (mode === "navigate") {
      setIndices(point);
      return;
    }
    if (mode === "trace") {
      if (!start) {
        setStart(point);
        return;
      }
      action(() => request(route + "/auto_trace", { start, end: point }));
      setStart(null);
    } else if (moving) {
      action(() =>
        request(
          `${base}/centerline/${side}/${moving.branch}/points/${moving.index}`,
          { point },
          "PUT",
        ),
      );
      setMoving(null);
    } else action(() => request(route + "/points", { point }));
  }
  function changeMode(value: string) {
    setMode(mode === value ? "navigate" : value);
    setStart(null);
    setMoving(null);
  }
  const count = points[side + ":" + branch]?.length || 0;
  return (
    <div className="workspace">
      <div className="workspace-main">
        <div className="viewer-layout">
          <aside className="workflow-column">{workflow}</aside>
          <div className="planes">
            {axes.map((axis, i) => (
              <SlicePane
                key={axis}
                axis={axis}
                index={indices[i]}
                study={study}
                base={base}
                wl={wl}
                ww={ww}
                points={points}
                side={side}
                start={start}
                indices={indices}
                onIndex={(n) =>
                  setIndices((previous) =>
                    previous.map((v, j) => (i === j ? n : v)),
                  )
                }
                onPoint={click}
              />
            ))}
          </div>
          <PathPreview
            points={points}
            paths={paths}
            spacing={study.spacing_mm}
            displayFlips={study.display_flips}
            origin={study.origin_mm}
            segmentations={wallSeries}
            side={side}
          />
        </div>
        <div className="control-rows">
          <div className="button-row">
            <div className="segmented" aria-label="Сторона">
              {[
                ["left", "Левая"],
                ["right", "Правая"],
              ].map(([key, label]) => (
                <button
                  key={key}
                  disabled={busy}
                  aria-pressed={side === key}
                  onClick={() => onSide(key)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="segmented" aria-label="Ветвь">
              {branches.map((b) => (
                <button
                  key={b}
                  disabled={busy}
                  aria-pressed={branch === b}
                  onClick={() => {
                    onBranch(b);
                    setStart(null);
                    setMoving(null);
                  }}
                >
                  CCA → {b === "cca_ica" ? "ICA" : "ECA"}
                </button>
              ))}
            </div>
            <button
              disabled={busy}
              aria-pressed={mode === "points"}
              onClick={() => changeMode("points")}
            >
              Разметить путь
            </button>
            <button
              disabled={busy}
              aria-pressed={mode === "trace"}
              onClick={() => changeMode("trace")}
            >
              Auto-Trace…
            </button>
            <button
              disabled={busy || !count}
              onClick={() => action(() => request(route, {}, "DELETE"))}
            >
              Очистить ветвь
            </button>
            <button
              className="primary"
              disabled={busy || count < 2}
              onClick={() => onStep(1)}
            >
              Контуры вдоль пути…
            </button>
            <span className="readout">
              {count}
              <small>точек</small>
            </span>
          </div>
          <p className="interaction-status" role="status">
            {busy
              ? "Обработка…"
              : moving
                ? "Нажмите новое положение точки на любом срезе"
                : start
                  ? "Выберите конец пути"
                  : mode === "trace"
                    ? "Выберите начало пути"
                    : mode === "points"
                      ? "Нажмите на сосуд, чтобы добавить точку"
                      : "Колесо — срезы · Правая кнопка + вверх/вниз — масштаб · Средняя кнопка — перемещение"}
          </p>
          <div className="button-row">
            <button className="primary" onClick={() => onStep(2)}>
              Поверхность…
            </button>
            <button onClick={() => onStep(3)}>Расчётная сетка…</button>
            <button onClick={() => onStep(4)}>Граничные условия…</button>
            <button onClick={() => onStep(5)}>CFD Computation…</button>
          </div>
          <div className="window-controls">
            <label>
              Level (WL): <output>{wl}</output>
              <input
                aria-label="Уровень окна"
                type="range"
                min={-1000}
                max={3000}
                value={wl}
                onChange={(e) => setWl(+e.target.value)}
              />
            </label>
            <label>
              Width (WW): <output>{ww}</output>
              <input
                aria-label="Ширина окна"
                type="range"
                min={1}
                max={6000}
                value={ww}
                onChange={(e) => setWw(+e.target.value)}
              />
            </label>
            <button
              onClick={() => {
                setWl(study.default_window.wl);
                setWw(study.default_window.ww);
              }}
            >
              Сбросить
            </button>
          </div>
        </div>
      </div>
      <aside className="points-panel">
        <h2 className="eyebrow">
          PATH POINTS — {side === "right" ? "RIGHT" : "LEFT"}
        </h2>
        <div className="point-columns">
          {branches.map((b) => (
            <div key={b}>
              <h3>
                <i style={{ background: branchColors[b] }} />
                CCA → {b === "cca_ica" ? "ICA" : "ECA"}
              </h3>
              {!points[side + ":" + b]?.length && <p className="muted">—</p>}
              {(points[side + ":" + b] || []).map((p, i) => (
                <div
                  className="point-card"
                  key={i}
                  style={{ borderColor: branchColors[b] + "80" }}
                >
                  <div className="point-card-heading">
                    <button
                      className="point-number"
                      title="Перейти к точке"
                      onClick={() => setIndices([...p])}
                    >
                      {i + 1}
                    </button>
                    <button
                      disabled={busy}
                      aria-label={`Удалить ${b} точку ${i + 1}`}
                      onClick={() =>
                        action(() =>
                          request(
                            `${base}/centerline/${side}/${b}/points/${i}`,
                            {},
                            "DELETE",
                          ),
                        )
                      }
                    >
                      ×
                    </button>
                  </div>
                  <code>{p.join(", ")}</code>
                  <div className="point-actions">
                    <button
                      disabled={busy}
                      title="Переместить точку"
                      onClick={() => {
                        onBranch(b);
                        setStart(null);
                        setMoving({ branch: b, index: i });
                        setMode("points");
                        setIndices([...p]);
                      }}
                    >
                      Изменить
                    </button>
                    <button
                      disabled={busy || i === 0}
                      aria-label={`Поднять ${b} точку ${i + 1}`}
                      onClick={() =>
                        action(() =>
                          request(
                            `${base}/centerline/${side}/${b}/reorder`,
                            { from_index: i, to_index: i - 1 },
                            "PUT",
                          ),
                        )
                      }
                    >
                      ↑
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
