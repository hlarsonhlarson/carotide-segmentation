import { useEffect, useMemo, useState } from "react";
import { request } from "./Segmentation";
import {
  labelVesselSegments,
  vesselSegmentNames,
  vesselSegments,
  type SegmentationPoint,
  type VesselSegment,
} from "./segmentationGeometry";

type Section = {
  index: number; distance_mm: number;
  lumen: { area_mm2: number; min_feret_mm: number } | null;
  reason?: string;
};
type Integral = {
  volume_mm3: number | null; covered_length_mm: number;
  excluded_intervals: { from_index: number; to_index: number; reason: string }[];
};
type Profile = {
  sections: Section[]; selected_length_mm: number;
  minimum_area_index: number | null; minimum_diameter_index: number | null;
  integrals: Record<IntegralKey, Integral>;
  density_parameters: { calcium_hu: number; min_component_mm2: number } | null;
  composition_parameters: { calcium_hu: number; lipid_hu: number; fat_hu: number } | null;
};
type IntegralKey = "lumen" | "annulus" | "high_density" | "plaque" | "calcified" | "lipid" | "fibrous";
const integralLabels: Record<IntegralKey, string> = {
  lumen: "Просвет", annulus: "Между просветом и стенкой", high_density: "Высокая плотность",
  plaque: "Бляшка/стенка (состав)", calcified: "· кальцинированная", lipid: "· атероматозная",
  fibrous: "· фиброзная",
};
const reasons: Record<string, string> = {
  different_composition_parameters: "Разные пороги состава бляшки",
  duplicate_position: "Совпадающие позиции",
  sampling_gap: "Большой шаг между сечениями",
  invalid_frame: "Некорректная плоскость",
  sharp_turn: "Резкий поворот пути",
  missing_measurement: "Нет измерения",
  different_density_parameters: "Разные или неизвестные параметры высокой плотности",
};

export function LongitudinalProfile({ route, points, index, onSelect, disabled, branch, otherPoints = [] }: {
  route: string; points: SegmentationPoint[]; index: number;
  onSelect: (index: number) => void; disabled: boolean;
  branch?: string; otherPoints?: SegmentationPoint[];
}) {
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(points.length - 1);
  const [maxStep, setMaxStep] = useState(4);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { setStart(0); setEnd(points.length - 1); }, [route, points.length]);
  // Index ranges of CCA / bifurcation / distal artery along this branch, so
  // lumen and plaque volumes can be read per vessel part.
  const segmentRanges = useMemo(() => {
    if (!branch || !otherPoints.length) return [];
    const isIca = branch === "cca_ica";
    const labels = labelVesselSegments(isIca ? points : otherPoints, isIca ? otherPoints : points);
    if (!labels.split) return [];
    const own = isIca ? labels.cca_ica : labels.cca_eca;
    return vesselSegments.flatMap((segment: VesselSegment) => {
      const indices = own.flatMap((label, i) => (label === segment ? [i] : []));
      return indices.length ? [{ segment, start: indices[0], end: indices[indices.length - 1] }] : [];
    });
  }, [branch, points, otherPoints]);
  useEffect(() => {
    let active = true;
    setProfile(null);
    setError("");
    if (!points.length || start > end || end >= points.length) {
      setBusy(false);
      return () => { active = false; };
    }
    setBusy(true);
    request(route + "/profile", { start_index: start, end_index: end, max_step_mm: maxStep })
      .then(value => { if (active) setProfile(value); })
      .catch(e => { if (active) setError(String(e)); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [route, points, start, end, maxStep]);
  const sections = profile?.sections || [];
  const areas = sections.flatMap(s => s.lumen ? [s.lumen.area_mm2] : []);
  const maxArea = Math.max(1, ...areas);
  const firstDistance = sections[0]?.distance_mm || 0;
  const x = (s: Section) => 35 + 510 * (s.distance_mm - firstDistance) / Math.max(1, profile?.selected_length_mm || 0);
  const y = (s: Section) => 145 - 120 * (s.lumen?.area_mm2 || 0) / maxArea;
  return <section aria-label="Профиль вдоль сосуда">
    <h3>Профиль просвета и оценка объёмов</h3>
    {!!segmentRanges.length && <div className="button-row" role="group" aria-label="Диапазон по отделу">
      {segmentRanges.map(r => <button key={r.segment} disabled={disabled}
        aria-pressed={start === r.start && end === r.end}
        onClick={() => { setStart(r.start); setEnd(r.end); }}>
        {vesselSegmentNames[r.segment]} ({r.start + 1}–{r.end + 1})
      </button>)}
      <button disabled={disabled} aria-pressed={start === 0 && end === points.length - 1}
        onClick={() => { setStart(0); setEnd(points.length - 1); }}>Весь путь</button>
    </div>}
    <label>Начальное сечение
      <input type="number" min={1} max={points.length} value={start + 1} disabled={disabled}
        onChange={e => setStart(Math.max(0, Math.min(points.length - 1, +e.target.value - 1)))} />
    </label>
    <label>Конечное сечение
      <input type="number" min={1} max={points.length} value={end + 1} disabled={disabled}
        onChange={e => setEnd(Math.max(0, Math.min(points.length - 1, +e.target.value - 1)))} />
    </label>
    <label>Максимальный шаг интегрирования, мм
      <input type="number" min={.5} max={10} step={.5} value={maxStep} disabled={disabled}
        onChange={e => setMaxStep(Math.max(.5, Math.min(10, +e.target.value || .5)))} />
    </label>
    {start > end && <p role="alert">Начало диапазона должно предшествовать концу.</p>}
    {busy && <p role="status">Расчёт профиля…</p>}
    {error && <p role="alert">{error}</p>}
    {profile && <>
      <svg viewBox="0 0 560 175" role="img" aria-label="Площадь просвета вдоль пути, мм²"
        style={{ width: "100%", minHeight: 150 }}>
        <path d="M35 20V145H545" fill="none" stroke="currentColor" />
        <text x={2} y={18} fill="currentColor" fontSize={12}>{maxArea.toFixed(1)} мм²</text>
        <text x={35} y={167} fill="currentColor" fontSize={12}>{firstDistance.toFixed(1)} мм</text>
        <text x={480} y={167} fill="currentColor" fontSize={12}>{sections.at(-1)?.distance_mm.toFixed(1)} мм</text>
        {sections.map((s, i) => {
          const previous = sections[i - 1];
          return s.lumen ? <g key={s.index}>
            {previous?.lumen && s.distance_mm - previous.distance_mm <= maxStep &&
              <line x1={x(previous)} y1={y(previous)} x2={x(s)} y2={y(s)} stroke="#26c6da" />}
            <circle cx={x(s)} cy={y(s)} r={s.index === index ? 5 : 3}
              fill={s.index === index ? "#ffca28" : "#26c6da"} role="button" tabIndex={disabled ? -1 : 0}
              aria-label={`Сечение ${s.index + 1}, площадь ${s.lumen.area_mm2.toFixed(2)} мм²`}
              onClick={() => { if (!disabled) onSelect(s.index); }}
              onKeyDown={e => { if (!disabled && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onSelect(s.index); } }}>
              <title>Сечение {s.index + 1}: {s.lumen.area_mm2.toFixed(2)} мм²</title>
            </circle>
          </g> : <path key={s.index} d={`M${x(s)-3} 140l6 10m-6 0l6 -10`} stroke="#ef5350" />;
        })}
      </svg>
      <div className="button-row">
        <button disabled={disabled || profile.minimum_area_index === null}
          onClick={() => onSelect(profile.minimum_area_index!)}>К минимуму площади</button>
        <button disabled={disabled || profile.minimum_diameter_index === null}
          onClick={() => onSelect(profile.minimum_diameter_index!)}>К минимуму диаметра Фере</button>
      </div>
      <p>Минимумы относятся к выбранному диапазону; это не автоматический выбор стеноза ICA.</p>
      {(Object.keys(integralLabels) as IntegralKey[]).map(key => {
        const value = profile.integrals[key];
        return <div key={key}>
          <strong>{integralLabels[key]}: </strong>
          {value.volume_mm3 === null ? "нет достаточных данных" : `${value.volume_mm3.toFixed(2)} мм³`}
          <span> · покрыто {value.covered_length_mm.toFixed(1)} из {profile.selected_length_mm.toFixed(1)} мм</span>
          {!!value.excluded_intervals.length && <details>
            <summary>Исключено интервалов: {value.excluded_intervals.length}</summary>
            <ul>{value.excluded_intervals.map(gap => <li key={gap.from_index}>
              <button disabled={disabled} onClick={() => onSelect(gap.from_index)}>
                {gap.from_index + 1}–{gap.to_index + 1}
              </button> {reasons[gap.reason] || gap.reason}
            </li>)}</ul>
          </details>}
        </div>;
      })}
      {profile.density_parameters && <p>Порог высокой плотности: {profile.density_parameters.calcium_hu} HU;
        минимальная площадь компонента: {profile.density_parameters.min_component_mm2} мм².</p>}
      {profile.composition_parameters && <p>Состав бляшки: атерома &lt; {profile.composition_parameters.lipid_hu} HU,
        кальций ≥ {profile.composition_parameters.calcium_hu} HU, жир &lt; {profile.composition_parameters.fat_hu} HU.</p>}
      <p>Приближённое интегрирование площадей, без заполнения пропусков.
        Не объём воксельной маски и не классификация бляшки. Области бифуркации и
        пересекающиеся сечения требуют проверки; объёмы перекрывающихся ветвей не суммируются.</p>
    </>}
  </section>;
}
