import { useEffect, useState } from "react";
import { request } from "./Segmentation";
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
export function PathEditor({
  base,
  side,
  branch,
  study,
  onError,
}: {
  base: string;
  side: string;
  branch: string;
  study: any;
  onError: (s: string) => void;
}) {
  const [axis, setAxis] = useState<Axis>("axial"),
    [index, setIndex] = useState(Math.floor(study.shape[0] / 2)),
    [points, setPoints] = useState<number[][]>([]),
    [busy, setBusy] = useState(false),
    [mode, setMode] = useState("navigate"),
    [start, setStart] = useState<number[] | null>(null),
    [moving, setMoving] = useState<number | null>(null),
    [wl, setWl] = useState(study.default_window.wl),
    [ww, setWw] = useState(study.default_window.ww);
  const route = `${base}/centerline/${side}/${branch}`,
    dimension = axis === "axial" ? 0 : axis === "coronal" ? 1 : 2,
    rows = axis === "axial" ? study.shape[1] : study.shape[0],
    cols = axis === "sagittal" ? study.shape[1] : study.shape[2];
  async function refresh() {
    setPoints(
      (await request(base + "/control_points"))[`${side}:${branch}`] || [],
    );
  }
  useEffect(() => {
    let active = true;
    request(base + "/control_points")
      .then((d) => {
        if (active) setPoints(d[`${side}:${branch}`] || []);
      })
      .catch((e) => onError(String(e)));
    return () => {
      active = false;
    };
  }, [route]);
  async function action(fn: () => Promise<unknown>) {
    setBusy(true);
    try {
      await fn();
      await refresh();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  }
  function click(e: React.MouseEvent<SVGSVGElement>) {
    if (busy || mode === "navigate") return;
    const r = e.currentTarget.getBoundingClientRect(),
      row = Math.min(
        rows - 1,
        Math.floor(((e.clientY - r.top) / r.height) * rows),
      ),
      col = Math.min(
        cols - 1,
        Math.floor(((e.clientX - r.left) / r.width) * cols),
      ),
      point = voxelAt(axis, index, row, col, study.shape);
    if (mode === "trace") {
      if (!start) {
        setStart(point);
        return;
      }
      action(() => request(route + "/auto_trace", { start, end: point }));
      setStart(null);
    } else if (moving !== null) {
      action(() => request(route + "/points/" + moving, { point }, "PUT"));
      setMoving(null);
    } else action(() => request(route + "/points", { point }));
  }
  return (
    <section>
      <h2>Путь сосуда</h2>
      <label>
        Плоскость
        <select
          value={axis}
          onChange={(e) => {
            const a = e.target.value as Axis;
            setAxis(a);
            setIndex(
              Math.floor(
                study.shape[a === "axial" ? 0 : a === "coronal" ? 1 : 2] / 2,
              ),
            );
          }}
        >
          <option value="axial">Аксиальная</option>
          <option value="coronal">Корональная</option>
          <option value="sagittal">Сагиттальная</option>
        </select>
      </label>
      <label>
        Режим
        <select
          value={mode}
          onChange={(e) => {
            setMode(e.target.value);
            setStart(null);
            setMoving(null);
          }}
        >
          <option value="navigate">Просмотр</option>
          <option value="points">Добавление точек</option>
          <option value="trace">Auto-Trace: начало и конец</option>
        </select>
      </label>
      <p>
        {start
          ? "Выберите конец пути"
          : moving !== null
            ? "Нажмите новое положение точки"
            : "Срез " + (index + 1) + " / " + study.shape[dimension]}
      </p>
      <div
        style={{
          position: "relative",
          width: "min(100%,600px)",
          aspectRatio: `${cols * study.spacing_mm[axis === "sagittal" ? 1 : 2]}/${rows * study.spacing_mm[axis === "axial" ? 1 : 0]}`,
        }}
      >
        <img
          style={{ width: "100%", height: "100%" }}
          src={
            "/api" +
            base +
            `/slice?axis=${axis}&index=${index}&wl=${wl}&ww=${ww}`
          }
          alt="DICOM-срез"
        />
        <svg
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
          }}
          viewBox={`0 0 ${cols} ${rows}`}
          preserveAspectRatio="none"
          onClick={click}
        >
          {points.map(
            (p, i) =>
              Math.abs(p[dimension] - index) < 1 && (
                <circle
                  key={i}
                  cx={(axis === "sagittal" ? p[1] : p[2]) + 0.5}
                  cy={
                    (axis === "axial" ? p[1] : study.shape[0] - 1 - p[0]) + 0.5
                  }
                  r={Math.max(cols / 100, 1)}
                  fill="#40ddc5"
                />
              ),
          )}
        </svg>
      </div>
      <input
        aria-label="Номер среза"
        type="range"
        min="0"
        max={study.shape[dimension] - 1}
        value={index}
        onChange={(e) => setIndex(+e.target.value)}
      />
      <label>
        Уровень окна
        <input
          type="number"
          value={wl}
          onChange={(e) => setWl(+e.target.value)}
        />
      </label>
      <label>
        Ширина окна
        <input
          type="number"
          min="1"
          value={ww}
          onChange={(e) => setWw(Math.max(1, +e.target.value))}
        />
      </label>
      <h3>Контрольные точки: {points.length}</h3>
      {points.map((p, i) => (
        <div className="point-row" key={i}>
          <span>
            {i + 1}. {p.join(", ")}
          </span>
          <button
            disabled={busy}
            onClick={() => {
              setMoving(i);
              setMode("points");
              setIndex(p[dimension]);
            }}
          >
            Переместить
          </button>
          <button
            disabled={busy || i === 0}
            onClick={() =>
              action(() =>
                request(
                  route + "/reorder",
                  { from_index: i, to_index: i - 1 },
                  "PUT",
                ),
              )
            }
          >
            ↑
          </button>
          <button
            disabled={busy}
            onClick={() =>
              action(() => request(route + "/points/" + i, {}, "DELETE"))
            }
          >
            Удалить
          </button>
        </div>
      ))}
    </section>
  );
}
