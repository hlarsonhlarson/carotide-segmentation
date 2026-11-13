import { useEffect, useRef, useState } from "react";
import { useImageNavigation } from "./useImageNavigation";
import { SegmentationPreview } from "./SegmentationPreview";
import { branchColors } from "./PathPreview";
import { LumenMeasurements } from "./LumenMeasurements";
import type {
  SegmentationPoint,
  SegmentationChange,
} from "./segmentationGeometry";
export async function request(path: string, body?: unknown, method = "POST") {
  const response = await fetch(
    "/api" + path,
    body === undefined
      ? {}
      : {
          method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    const error = await response.json();
    throw Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}
export function Segmentation({
  base,
  side,
  branch,
  study,
  focusIndex = null,
  onSeriesChange,
  onError,
}: {
  base: string;
  side: string;
  branch: string;
  study: any;
  focusIndex?: number | null;
  onSeriesChange: SegmentationChange;
  onError: (message: string) => void;
}) {
  const latestPoints = useRef<SegmentationPoint[]>([]);
  const [points, setPoints] = useState<SegmentationPoint[]>([]);
  const [index, setIndex] = useState(0);
  const [method, setMethod] = useState("auto");
  const isManual = method === "manual" || method === "wall_manual";
  const [radius, setRadius] = useState(3);
  const [calciumHu, setCalciumHu] = useState(600);
  const [wallSearch, setWallSearch] = useState(6);
  const [manual, setManual] = useState<number[][]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{
    done: number;
    total: number;
  } | null>(null);
  const [wl, setWl] = useState<number>(
    study.modality === "CT" ? 150 : study.default_window.wl,
  );
  const [ww, setWw] = useState<number>(
    study.modality === "CT" ? 350 : study.default_window.ww,
  );
  const lock = useRef(false);
  const navigation = useImageNavigation(160, 160);
  const content = useRef<SVGGElement>(null);
  const route = `${base}/segment/${side}/${branch}`;
  const color = branchColors[branch];
  function publish(next: SegmentationPoint[]) {
    latestPoints.current = next;
    setPoints(next);
    onSeriesChange(base, side, branch, next);
  }
  async function refresh() {
    const data = await request(route);
    publish(data.points);
    setIndex((i) => Math.min(i, Math.max(0, data.points.length - 1)));
  }
  useEffect(() => {
    let active = true;
    request(route)
      .then((d) => {
        if (active) publish(d.points);
      })
      .catch((e) => {
        if (active) onError(String(e));
      });
    return () => {
      active = false;
    };
  }, [route]);
  useEffect(() => {
    if (focusIndex === null || !points.length) return;
    setIndex(Math.max(0, Math.min(focusIndex, points.length - 1)));
    setManual([]);
  }, [focusIndex, points.length]);
  async function action(fn: () => Promise<unknown>) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    onError("");
    try {
      await fn();
    } catch (e) {
      onError(String(e));
    } finally {
      // Also refresh after a partially successful batch so the preview matches
      // the accepted server state when a later section could not be segmented.
      try {
        await refresh();
      } catch (e) {
        onError(String(e));
      }
      lock.current = false;
      setBusy(false);
      setProgress(null);
    }
  }
  function goTo(next: number) {
    setIndex(Math.max(0, Math.min(points.length - 1, next)));
    setManual([]);
  }
  async function segmentAt(i: number) {
    if (method === "wall_manual") {
      await request(route + `/point/${i}/wall`, {
        calcium_hu: calciumHu, wall_contour_px: manual,
      });
      setManual([]);
      return;
    }
    const result = await request(
      route + `/point/${i}` + (method === "manual" ? "/manual" : ""),
      method === "manual"
        ? { contour_px: manual }
        : { method, radius_mm: radius },
    );
    publish(
      latestPoints.current.map((p, j) =>
        j === i
          ? {
              ...p,
              wall_analysis: undefined,
              contour_mm: result.contour_mm,
              contour_px: result.contour_px,
              radius_mm: result.fitted_radius_mm,
            }
          : p,
      ),
    );
  }
  const accepted = points.filter((p) => p.contour_px?.length).length;
  return (
    <section className="segmentation-workspace">
      <div className="segmentation-editor">
        <div className="segmentation-title">
          <h2>Контуры · CCA → {branch === "cca_ica" ? "ICA" : "ECA"}</h2>
          <button
            disabled={busy}
            onClick={() =>
              action(async () => {
                await request(route + "/prepare", {});
                setIndex(0);
                setManual([]);
              })
            }
          >
            {points.length
              ? "Подготовить сечения заново"
              : "Подготовить сечения"}
          </button>
        </div>
        {!points.length ? (
          <p className="segmentation-empty">
            Подготовьте поперечные сечения по размеченному пути. Справа появятся
            линия сосуда и положение текущего сечения.
          </p>
        ) : (
          <>
            <div className="section-position">
              <span>
                Сечение {index + 1} / {points.length}
              </span>
              <span>
                {points[index]?.contour_px?.length
                  ? "✓ Контур готов"
                  : "Контур не задан"}
              </span>
            </div>
            <div className="segmentation-slice">
              <svg
                {...navigation.handlers}
                viewBox="0 0 160 160"
                aria-label="Редактор поперечного сечения"
                onClick={(e) => {
                  if (
                    !isManual ||
                    busy ||
                    e.button !== 0 ||
                    !navigation.canClick()
                  )
                    return;
                  const matrix = content.current?.getScreenCTM();
                  if (!matrix) return;
                  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(
                    matrix.inverse(),
                  );
                  if (p.x >= 0 && p.x < 160 && p.y >= 0 && p.y < 160)
                    setManual((previous) => [...previous, [p.y, p.x]]);
                }}
              >
                <g ref={content} transform={navigation.transform}>
                  <image
                    href={`/api${route}/slice/${index}?wl=${wl}&ww=${ww}`}
                    width={160}
                    height={160}
                    preserveAspectRatio="none"
                    aria-label="Поперечное сечение сосуда"
                  />
                  <polygon
                    points={(points[index]?.contour_px || [])
                      .map((p) => `${p[1]},${p[0]}`)
                      .join(" ")}
                    fill={color + "22"}
                    stroke={color}
                    strokeWidth={0.7 / navigation.zoom}
                  />
                  {isManual && (
                    <>
                      <polyline
                        points={manual.map((p) => `${p[1]},${p[0]}`).join(" ")}
                        fill="none"
                        stroke="#fff"
                        strokeWidth={0.6 / navigation.zoom}
                        strokeDasharray="2 1"
                      />
                      {manual.map((p, i) => (
                        <circle
                          key={i}
                          cx={p[1]}
                          cy={p[0]}
                          r={0.8 / navigation.zoom}
                          fill="#fff"
                        />
                      ))}
                    </>
                  )}
                  {points[index]?.wall_analysis && <g pointerEvents="none">
                    {points[index].wall_analysis!.high_density_pixels.map(([r, c], i) =>
                      <rect key={i} x={c - .5} y={r - .5} width={1} height={1} fill="#ffca28" opacity={.65} />)}
                    <polygon points={points[index].wall_analysis!.wall_contour_px.map(([r, c]) => `${c},${r}`).join(" ")}
                      fill="none" stroke="#ff80ab" strokeWidth={.7 / navigation.zoom} />
                  </g>}
                </g>
              </svg>
            </div>
            <div className="section-navigation">
              <button
                disabled={busy || index === 0}
                onClick={() => goTo(index - 1)}
              >
                ◀ Назад
              </button>
              <input
                aria-label="Сечение вдоль пути"
                type="range"
                min={0}
                max={points.length - 1}
                value={index}
                disabled={busy}
                onChange={(e) => goTo(+e.target.value)}
              />
              <button
                disabled={busy || index === points.length - 1}
                onClick={() => goTo(index + 1)}
              >
                Далее ▶
              </button>
            </div>
            <div className="button-row">
              <button onClick={navigation.reset}>Вписать сечение</button>
              <small>Правая кнопка — масштаб · Средняя — перемещение</small>
            </div>
            <div className="segment-contrast">
              <label>
                Level (WL): {wl}
                <input
                  aria-label="Контраст сечения: уровень"
                  type="range"
                  min={-1000}
                  max={Math.max(3000, study.default_window.wl * 2)}
                  value={wl}
                  onChange={(e) => setWl(+e.target.value)}
                />
              </label>
              <label>
                Width (WW): {ww}
                <input
                  aria-label="Контраст сечения: ширина"
                  type="range"
                  min={1}
                  max={Math.max(6000, study.default_window.ww * 2)}
                  value={ww}
                  onChange={(e) => setWw(+e.target.value)}
                />
              </label>
              <button
                onClick={() => {
                  setWl(
                    study.modality === "CT" ? 150 : study.default_window.wl,
                  );
                  setWw(
                    study.modality === "CT" ? 350 : study.default_window.ww,
                  );
                }}
              >
                Сбросить контраст
              </button>
            </div>
            <div className="segment-parameters">
              <label>
                Метод
                <select
                  aria-label="Метод"
                  value={method}
                  disabled={busy}
                  onChange={(e) => {
                    setMethod(e.target.value);
                    setManual([]);
                  }}
                >
                  {[
                    ["auto", "Автоматически"],
                    ["threshold", "Порог"],
                    ["fwhm", "Просвет по HU-профилю (КТА)"],
                    ["levelset", "Активный контур"],
                    ["circle", "Круг"],
                    ["manual", "Ручной контур"],
                    ...(study.modality === "CT" ? [["wall_manual", "Ручная внешняя стенка"]] : []),
                  ].map(([v, t]) => (
                    <option key={v} value={v}>
                      {t}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Начальный радиус, мм
                <input
                  type="number"
                  min={0.1}
                  max={50}
                  step={0.1}
                  value={radius}
                  disabled={busy || isManual}
                  onChange={(e) =>
                    setRadius(
                      Math.max(0.1, Math.min(50, +e.target.value || 0.1)),
                    )
                  }
                />
              </label>
            </div>
            <div className="button-row">
              {isManual && (
                <button
                  disabled={busy || !manual.length}
                  onClick={() => setManual((p) => p.slice(0, -1))}
                >
                  Убрать последнюю точку
                </button>
              )}
              <button
                className="primary"
                disabled={busy || (isManual && manual.length < 3) ||
                  (method === "wall_manual" && !points[index]?.contour_px?.length)}
                onClick={() =>
                  action(async () => {
                    await segmentAt(index);
                    setManual([]);
                  })
                }
              >
                Применить к сечению
              </button>
              <button
                disabled={busy || !points[index]?.contour_px?.length}
                onClick={() =>
                  action(async () => {
                    await request(route + `/point/${index}`, {}, "DELETE");
                    setManual([]);
                  })
                }
              >
                Удалить контур
              </button>
              <button
                disabled={busy || isManual}
                onClick={() =>
                  action(async () => {
                    setProgress({ done: 0, total: points.length });
                    for (let i = 0; i < points.length; i++) {
                      await segmentAt(i);
                      setProgress({ done: i + 1, total: points.length });
                    }
                  })
                }
              >
                Обработать все сечения
              </button>
            </div>
            {isManual && (
              <p>
                Отметьте точки по границе сосуда, затем примените контур. Белый
                пунктир показывает несохранённый рисунок.
              </p>
            )}
            {progress && (
              <progress
                aria-label="Обработка сечений"
                max={progress.total}
                value={progress.done}
              />
            )}
            <p role="status">
              {accepted} из {points.length} контуров
              {progress
                ? ` · Обработка ${progress.done} / ${progress.total}`
                : busy
                  ? " · Обработка…"
                  : ""}
            </p>
            <LumenMeasurements route={route} branch={branch} index={index} points={points} />
            {study.modality === "CT" && <section aria-label="Стенка и кальций">
              <h3>Внешняя стенка · экспериментальный контур</h3>
              <label>Порог высокой плотности, HU
                <input type="number" min={131} max={3000} value={calciumHu}
                  onChange={e => setCalciumHu(+e.target.value)} />
              </label>
              <label>Поиск за просветом, мм
                <input type="number" min={.5} max={6} step={.2} value={wallSearch}
                  onChange={e => setWallSearch(+e.target.value)} />
              </label>
              <button disabled={busy || !points[index]?.contour_px?.length || points[index]?.method === "circle"}
                onClick={() => action(() => request(route + `/point/${index}/wall`,
                  { calcium_hu: calciumHu, max_thickness_mm: wallSearch }))}>
                Найти внешнюю границу
              </button>
              <p>Розовый — предложение границы; жёлтый — высокая плотность.
                Порог выбирается с учётом контрастирования КТА; это не классификация бляшки.</p>
              {points[index]?.wall_analysis && <p>
                Площадь между контурами: {points[index].wall_analysis!.annulus_area_mm2.toFixed(2)} мм² ·
                Высокая плотность: {points[index].wall_analysis!.high_density_area_mm2.toFixed(2)} мм²
                (≥{points[index].wall_analysis!.parameters.calcium_hu} HU) ·
                {points[index].wall_analysis!.max_radial_thickness_mm != null &&
                  <>Максимальная радиальная толщина: {points[index].wall_analysis!.max_radial_thickness_mm!.toFixed(2)} мм · </>}
                {points[index].wall_analysis!.edge_support_fraction != null &&
                  <>Поддержка границы: {(points[index].wall_analysis!.edge_support_fraction! * 100).toFixed(0)}%. </>}
                {points[index].wall_analysis!.status === "manual" && "Внешняя стенка размечена вручную. "}
                Только это сечение; результат требует проверки.
              </p>}
            </section>}
          </>
        )}
      </div>
      <SegmentationPreview
        points={points}
        index={index}
        radius={radius}
        branch={branch}
        origin={study.origin_mm}
        displayFlips={study.display_flips}
      />
    </section>
  );
}
