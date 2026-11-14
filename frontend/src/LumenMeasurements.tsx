import { useEffect, useState } from "react";
import { request } from "./Segmentation";
import type { SegmentationPoint } from "./segmentationGeometry";

type Metrics = {
  area_mm2: number;
  equivalent_diameter_mm: number;
  min_feret_mm: number;
  max_feret_mm: number;
};
type Result = {
  section: Metrics;
  nascet: { percent: number } | null;
};

export function LumenMeasurements({
  route, branch, index, points,
}: {
  route: string; branch: string; index: number; points: SegmentationPoint[];
}) {
  const [reference, setReference] = useState<number | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setReference(null);
    setConfirmed(false);
  }, [route]);
  // A response from old geometry or a previous section must never remain on
  // screen after edits. Measurements are read-only and can always be repeated.
  useEffect(() => {
    let active = true;
    setResult(null);
    setError("");
    if (!points[index]?.contour_mm?.length) {
      setBusy(false);
      return () => { active = false; };
    }
    setBusy(true);
    request(route + "/measurements", {
      index,
      reference_index: confirmed ? reference : null,
      confirmed_ica_reference: confirmed,
    })
      .then((value) => { if (active) setResult(value); })
      .catch((e) => { if (active) setError(String(e)); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [route, points, index, reference, confirmed]);
  return (
    <section aria-label="Измерения просвета" className="lumen-measurements">
      <h3>Измерения просвета</h3>
      {busy && <p role="status">Измерение…</p>}
      {!points[index]?.contour_mm?.length && <p>Сначала разметьте просвет этого сечения.</p>}
      {result && <p>
        Площадь: {result.section.area_mm2.toFixed(2)} мм² ·
        Эквивалентный диаметр: {result.section.equivalent_diameter_mm.toFixed(2)} мм ·
        Диаметры Фере: {result.section.min_feret_mm.toFixed(2)}–{result.section.max_feret_mm.toFixed(2)} мм
      </p>}
      {branch === "cca_ica" && <>
        <button disabled={!points[index]?.contour_mm?.length || points[index]?.method === "circle"}
          onClick={() => { setReference(index); setConfirmed(false); }}>
          Выбрать текущее сечение как дистальный референс ICA
        </button>
        {reference !== null && <>
          <p>Референс: сечение {reference + 1}. Перейдите к сужению проксимальнее него.</p>
          <label>
            <input type="checkbox" checked={confirmed} disabled={index >= reference}
              onChange={(e) => setConfirmed(e.target.checked)} />
            Сужение находится в ICA; референс — нормальная дистальная ICA,
            без признаков near-occlusion
          </label>
          <button onClick={() => { setReference(null); setConfirmed(false); }}>Сбросить референс</button>
        </>}
        {result?.nascet && <p role="status">
          Оценка NASCET: {result.nascet.percent.toFixed(1)}%.
          По минимальным диаметрам Фере; метод не валидирован клинически.
        </p>}
      </>}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
