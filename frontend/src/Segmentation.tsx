import { useEffect, useRef, useState } from "react";
import { useImageNavigation } from "./useImageNavigation";
import { SegmentationPreview } from "./SegmentationPreview";
import { branchColors } from "./PathPreview";
import { LumenMeasurements } from "./LumenMeasurements";
import { LongitudinalProfile } from "./LongitudinalProfile";
import type {
  PlaqueComposition,
  SegmentationPoint,
  SegmentationChange,
} from "./segmentationGeometry";
export class RequestError extends Error {
  constructor(message: string, public status: number) { super(message); }
}
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
    throw new RequestError(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail), response.status,
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
  const [lipidHu, setLipidHu] = useState(50);
  const [fatHu, setFatHu] = useState(-50);
  const [showComposition, setShowComposition] = useState(true);
  const wallParams = { calcium_hu: calciumHu, lipid_hu: lipidHu, fat_hu: fatHu };
  const stopWallBatch = useRef(false);
  const [wallBatch, setWallBatch] = useState(false);
  const [wallFailures, setWallFailures] = useState<{ index: number; reason: string }[]>([]);
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
  useEffect(() => {
    setWallFailures([]);
    return () => { stopWallBatch.current = true; };
  }, [route]);
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
        ...wallParams, wall_contour_px: manual,
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
  async function fillMissingWalls() {
    const candidates = points.flatMap((p, i) =>
      p.contour_px?.length && p.method !== "circle" && !p.wall_analysis ? [i] : []);
    const failures: { index: number; reason: string }[] = [];
    stopWallBatch.current = false;
    setWallBatch(true);
    setWallFailures([]);
    setProgress({ done: 0, total: candidates.length });
    try {
      for (let j = 0; j < candidates.length; j++) {
        if (stopWallBatch.current) break;
        const i = candidates[j];
        try {
          await request(route + `/point/${i}/wall`, {
            ...wallParams, max_thickness_mm: wallSearch,
          });
        } catch (e) {
          if (!(e instanceof RequestError) || e.status !== 422) throw e;
          failures.push({ index: i, reason: e.message });
          setWallFailures([...failures]);
        }
        setProgress({ done: j + 1, total: candidates.length });
      }
    } finally {
      setWallBatch(false);
    }
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
                    {showComposition && points[index].wall_analysis!.composition
                      ? <PlaqueOverlay composition={points[index].wall_analysis!.composition!} />
                      : points[index].wall_analysis!.high_density_pixels.map(([r, c], i) =>
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
              <label>Кальций от, HU (не ниже 1,3 × HU просвета)
                <input type="number" min={131} max={3000} value={calciumHu}
                  onChange={e => setCalciumHu(+e.target.value)} />
              </label>
              <label>Атероматозный компонент ниже, HU
                <input type="number" min={-100} max={130} value={lipidHu}
                  onChange={e => setLipidHu(+e.target.value)} />
              </label>
              <label>Жир и воздух ниже, HU
                <input type="number" min={-300} max={129} value={fatHu}
                  onChange={e => setFatHu(+e.target.value)} />
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={showComposition}
                  onChange={e => setShowComposition(e.target.checked)} /> Показывать состав бляшки
              </label>
              <label>Поиск за просветом, мм
                <input type="number" min={.5} max={6} step={.2} value={wallSearch}
                  onChange={e => setWallSearch(+e.target.value)} />
              </label>
              <button disabled={busy || !points[index]?.contour_px?.length || points[index]?.method === "circle"}
                onClick={() => action(() => request(route + `/point/${index}/wall`,
                  { ...wallParams, max_thickness_mm: wallSearch }))}>
                Найти внешнюю границу
              </button>
              <button disabled={busy || !points.some(p => p.contour_px?.length && p.method !== "circle" && !p.wall_analysis)}
                onClick={() => action(fillMissingWalls)}>
                Найти стенку на незаполненных сечениях
              </button>
              <button disabled={busy || !points[index]?.wall_analysis}
                onClick={() => action(() => request(route + `/point/${index}/wall`, {}, "DELETE"))}>
                Удалить только внешнюю стенку
              </button>
              {wallBatch && <button onClick={() => { stopWallBatch.current = true; }}>
                Остановить после текущего сечения
              </button>}
              {!!wallFailures.length && <details>
                <summary>Стенка не найдена: {wallFailures.length} сечений</summary>
                <ul>{wallFailures.map(failure => <li key={failure.index}>
                  <button disabled={busy} onClick={() => goTo(failure.index)}>Сечение {failure.index + 1}</button>
                  {" "}{failure.reason}
                </li>)}</ul>
              </details>}
              <p>Розовый — внешняя граница. Состав бляшки: <span style={{ color: PLAQUE_COLORS.calcified }}>■</span> кальцинированный
                (≥ порога высокой плотности), <span style={{ color: PLAQUE_COLORS.lipid }}>■</span> атероматозный
                (низкая плотность), <span style={{ color: PLAQUE_COLORS.fibrous }}>■</span> фиброзный (остальное).
                Классификация по HU на КТА экспериментальна: кальций «расплывается», а контраст
                частично заходит в низкие значения. Результат требует проверки.</p>
              {points[index]?.wall_analysis && !points[index].wall_analysis!.composition &&
                <p>Состав для этого сечения не рассчитан: стенка найдена старой версией. Найдите её заново.</p>}
              {points[index]?.wall_analysis?.composition && <CompositionSummary
                composition={points[index].wall_analysis!.composition!} />}
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
            <LongitudinalProfile route={route} points={points} index={index}
              onSelect={goTo} disabled={busy} />
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

const PLAQUE_COLORS = { calcified: "#ef5350", lipid: "#ffd54f", fibrous: "#66bb6a" };

function pixelPath(pixels: number[][]) {
  return pixels.map(([r, c]) => `M${c - .5},${r - .5}h1v1h-1z`).join("");
}

function PlaqueOverlay({ composition }: { composition: PlaqueComposition }) {
  return <>
    <path d={pixelPath(composition.fibrous_pixels)} fill={PLAQUE_COLORS.fibrous} opacity={.35} />
    <path d={pixelPath(composition.lipid_pixels)} fill={PLAQUE_COLORS.lipid} opacity={.7} />
    <path d={pixelPath(composition.calcified_pixels)} fill={PLAQUE_COLORS.calcified} opacity={.85} />
  </>;
}

function CompositionSummary({ composition: c }: { composition: PlaqueComposition }) {
  const share = (value: number) => c.plaque_area_mm2 > 0 ? ` (${(100 * value / c.plaque_area_mm2).toFixed(0)}%)` : "";
  return <p>
    Бляшка/стенка: {c.plaque_area_mm2.toFixed(2)} мм² ·
    кальцинированная: {c.calcified_area_mm2.toFixed(2)} мм²{share(c.calcified_area_mm2)} ·
    атероматозная: {c.lipid_area_mm2.toFixed(2)} мм²{share(c.lipid_area_mm2)} ·
    фиброзная: {c.fibrous_area_mm2.toFixed(2)} мм²{share(c.fibrous_area_mm2)}.
    {c.calcium_in_lumen_area_mm2 > 0 &&
      <> Кальций внутри контура просвета: {c.calcium_in_lumen_area_mm2.toFixed(2)} мм² — площадь просвета завышена. </>}
    {c.contrast_partial_volume_area_mm2 > 0 &&
      <> Размытый край контраста (≥{c.parameters.fibrous_max_hu} HU, не бляшка): {c.contrast_partial_volume_area_mm2.toFixed(2)} мм². </>}
    {c.excluded_low_hu_area_mm2 > 0 &&
      <> Ниже {c.parameters.fat_hu} HU (жир/воздух, не бляшка): {c.excluded_low_hu_area_mm2.toFixed(2)} мм². </>}
    {" "}Пороги: атерома &lt; {c.parameters.lipid_hu} HU, фиброзная до {c.parameters.fibrous_max_hu} HU,
    кальций ≥ {c.calcium_hu_effective.toFixed(0)} HU{c.lumen_reference_hu != null &&
      ` (просвет ≈ ${c.lumen_reference_hu.toFixed(0)} HU)`};
    атерома не ищется ближе {c.parameters.edge_margin_mm} мм к внешней границе.
  </p>;
}
