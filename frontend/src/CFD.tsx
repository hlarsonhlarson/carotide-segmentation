import { useEffect, useMemo, useRef, useState } from "react";
import { ResultMesh, type Streamlines } from "./ResultMesh";

const BLOOD_VISCOSITY_PAS = 0.0035;
// Field -> label and the default colour range. "auto" = 2nd-98th percentile
// (single wall nodes at the caps otherwise stretch the scale); fixed ranges
// follow published carotid CFD figures (shear rate 0-2000 1/s, OSI 0-0.5).
const FIELDS: Record<string, { label: string; range: "auto" | [number, number] }> = {
  tawss_pa: { label: "TAWSS, Па", range: "auto" },
  shear_rate: { label: "Скорость сдвига (TAWSS/μ), 1/с", range: [0, 2000] },
  osi: { label: "OSI", range: [0, 0.5] },
  rrt_pa_inv: { label: "RRT, 1/Па", range: "auto" },
  mean_pressure_mmhg: { label: "Среднее давление, мм рт. ст.", range: "auto" },
  wss: { label: "WSS в момент времени, Па", range: "auto" },
  pressure: { label: "Давление в момент времени, Па", range: "auto" },
};
function percentile(values: number[], q: number) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return 0;
  return v[Math.min(v.length - 1, Math.max(0, Math.round(q * (v.length - 1))))];
}
// Percentiles of a large sample (all frames of a cycle) from a strided subset.
function sampled(values: ArrayLike<number>, max = 200000) {
  const stride = Math.max(1, Math.floor(values.length / max));
  const out: number[] = [];
  for (let i = 0; i < values.length; i += stride) out.push(values[i]);
  return out;
}
function niceRange(values: ArrayLike<number>): [number, number] {
  const v = sampled(values);
  const lo = percentile(v, 0.02), hi = percentile(v, 0.98);
  const step = Math.pow(10, Math.floor(Math.log10(Math.max(hi - lo, 1e-9)))) / 2;
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}
import { request } from "./Segmentation";
type Params = {
  psv_cms: number;
  edv_cms: number;
  cardiac_cycle_s: number;
  ica_fraction: number;
  ica_map_mmhg: number;
  eca_map_mmhg: number;
};
const labels: Record<keyof Params, string> = {
  psv_cms: "Пиковая скорость, см/с",
  edv_cms: "Диастолическая скорость, см/с",
  cardiac_cycle_s: "Сердечный цикл, с",
  ica_fraction: "Доля потока ICA",
  ica_map_mmhg: "Давление ICA, мм рт. ст.",
  eca_map_mmhg: "Давление ECA, мм рт. ст.",
};
export function CFD({
  base,
  side,
  onError,
}: {
  base: string;
  side: string;
  onError: (s: string) => void;
}) {
  const route = `${base}/segment/${side}`;
  const [params, setParams] = useState<Params | null>(null),
    [status, setStatus] = useState<any>({ state: "idle" }),
    [post, setPost] = useState("idle"),
    [errors, setErrors] = useState<string[]>([]),
    [busy, setBusy] = useState(false),
    [result, setResult] = useState<any>(null),
    [field, setField] = useState("tawss_pa"),
    [frame, setFrame] = useState(0),
    [frames, setFrames] = useState<{ field: string; data: Float32Array; points: number } | null>(null),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(0.25),
    [wholeRun, setWholeRun] = useState(false),
    [manualRange, setManualRange] = useState<[number, number] | null>(null),
    [showStreamlines, setShowStreamlines] = useState(false),
    [streamlines, setStreamlines] = useState<Streamlines | null>(null);
  const timeResolved = field === "wss" || field === "pressure";
  // All frames of a time-resolved field in one binary request, so playback
  // only recolours the mesh.
  useEffect(() => {
    let active = true;
    setFrames(null);
    setPlaying(false);
    if (result && timeResolved)
      fetch("/api" + route + "/su2_postprocess/frames?field=" + field)
        .then(async (r) => {
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
          const data = new Float32Array(await r.arrayBuffer());
          if (active) setFrames({ field, data, points: Number(r.headers.get("X-Points")) || result.points_mm.length });
        })
        .catch((e) => { if (active) onError(String(e)); });
    return () => { active = false; };
  }, [route, result, field]);
  // Playback runs over the settled (last) cardiac cycle unless asked otherwise.
  const frameIndices = useMemo<number[]>(() => {
    if (!result) return [];
    const times: number[] = result.times_s;
    const all = times.map((_, i) => i);
    const cycle = params?.cardiac_cycle_s;
    if (wholeRun || !cycle || !times.length) return all;
    // Exactly one cycle ending at the last recorded frame.
    const end = times[times.length - 1], dt = times.length > 1 ? end - times[times.length - 2] : 0;
    const last = all.filter((i) => times[i] > end - cycle + dt / 2);
    return last.length ? last : all;
  }, [result, wholeRun, params?.cardiac_cycle_s]);
  const frameValues = useMemo(() => frames && frames.field === field
    ? frames.data.subarray(frame * frames.points, (frame + 1) * frames.points) : null, [frames, field, frame]);
  useEffect(() => {
    if (frameIndices.length && !frameIndices.includes(frame)) setFrame(frameIndices[0]);
  }, [frameIndices]);
  // Playback follows simulated time (frames may be recorded unevenly),
  // slowed down by `speed`; the shown frame is the last one at or before it.
  const playRef = useRef({ frameIndices, frame });
  playRef.current = { frameIndices, frame };
  useEffect(() => {
    if (!playing || !result) return;
    const times: number[] = result.times_s;
    const idx = playRef.current.frameIndices;
    if (idx.length < 2) return;
    const t0 = times[idx[0]], tn = times[idx[idx.length - 1]];
    const span = tn - t0 + (tn - times[idx[idx.length - 2]]);
    const startSim = times[playRef.current.frame] ?? t0;
    const startWall = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const sim = t0 + ((startSim - t0 + ((now - startWall) / 1000) * speed) % span);
      let lo = 0, hi = idx.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (times[idx[mid]] <= sim + 1e-9) lo = mid; else hi = mid - 1;
      }
      if (idx[lo] !== playRef.current.frame) setFrame(idx[lo]);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, result, frameIndices]);
  // One colour scale for the whole played interval, so colours do not jump.
  const cycleRange = useMemo<[number, number] | null>(() => {
    if (!frames || frames.field !== field || !frameIndices.length) return null;
    const a = frameIndices[0], b = frameIndices[frameIndices.length - 1] + 1;
    return niceRange(frames.data.subarray(a * frames.points, b * frames.points));
  }, [frames, field, frameIndices]);
  const values: ArrayLike<number> = useMemo(() => result
    ? timeResolved
      ? frameValues || []
      : field === "shear_rate"
        ? result.tawss_pa.map((v: number) => v / BLOOD_VISCOSITY_PAS)
        : field === "mean_pressure_mmhg"
          ? result.mean_pressure_pa.map((v: number) => v / 133.322)
          : result[field]
    : [], [result, field, frameValues]);
  const preset = FIELDS[field]?.range ?? "auto";
  const autoRange = useMemo<[number, number]>(
    () => (preset !== "auto" ? preset : timeResolved ? cycleRange ?? [0, 1] : niceRange(values)),
    [timeResolved ? cycleRange : values, preset, timeResolved]);
  const colourRange = useMemo<[number, number]>(() => manualRange ?? autoRange, [manualRange, autoRange]);
  const streamRange = useMemo<[number, number] | undefined>(
    () => (streamlines ? [0, niceRange(streamlines.speed_cms)[1]] : undefined), [streamlines]);
  useEffect(() => { setManualRange(null); }, [field]);
  // Finished results open by themselves; no separate "open" step.
  useEffect(() => {
    let active = true;
    if (post === "done" && !result)
      request(route + "/su2_postprocess/result")
        .then((r) => { if (active) setResult(r); })
        .catch((e) => { if (active) onError(String(e)); });
    return () => { active = false; };
  }, [post, result, route]);
  useEffect(() => {
    let active = true;
    if (showStreamlines && result && !streamlines)
      request(route + "/su2_postprocess/streamlines")
        .then((d) => { if (active) setStreamlines(d); })
        .catch((e) => { if (active) { onError(String(e)); setShowStreamlines(false); } });
    return () => { active = false; };
  }, [showStreamlines, result, route]);
  useEffect(() => {
    let active = true;
    setResult(null);
    setFrame(0);
    setPost("idle");
    request(route + "/su2_params")
      .then((p) => {
        if (active) setParams(p);
      })
      .catch((e) => onError(String(e)));
    request("/health")
      .then((h) => {
        if (active) setErrors(h.solver_errors);
      })
      .catch((e) => onError(String(e)));
    async function poll() {
      try {
        const [s, p] = await Promise.all([
          request(route + "/su2_run/status"),
          request(route + "/su2_postprocess/status"),
        ]);
        if (active) {
          setStatus(s);
          setPost(p.state);
          if (p.state !== "done") setResult(null);
        }
      } catch (e) {
        if (active) onError(String(e));
      }
    }
    poll();
    const timer = setInterval(poll, 1500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [route]);
  async function action(fn: () => Promise<unknown>) {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section>
      <h2>Граничные условия и CFD</h2>
      {params &&
        Object.keys(labels).map((k) => {
          const key = k as keyof Params;
          return (
            <label key={key}>
              {labels[key]}
              <input
                type="number"
                step="any"
                value={params[key]}
                disabled={busy || status.state === "running"}
                onChange={(e) =>
                  setParams({ ...params, [key]: +e.target.value })
                }
              />
            </label>
          );
        })}
      <button
        disabled={!params || busy}
        onClick={() => action(() => request(route + "/su2_params", params))}
      >
        Сохранить условия
      </button>
      {errors.length > 0 && (
        <p>
          Расчёт недоступен: совместимая сборка SU2 ещё не подключена. Разметка,
          геометрия и просмотр сохранённых результатов доступны.
        </p>
      )}
      <button
        disabled={
          busy || errors.length > 0 || !params || status.state === "running"
        }
        onClick={() =>
          action(async () => {
            await request(route + "/su2_params", params);
            await request(route + "/su2_run", {});
            setStatus({ state: "running" });
          })
        }
      >
        Запустить CFD
      </button>
      <button
        disabled={status.state !== "running" || busy}
        onClick={() => action(() => request(route + "/su2_run/cancel", {}))}
      >
        Остановить расчёт
      </button>
      <p role="status">
        Расчёт: {status.state}{" "}
        {status.step !== undefined &&
          `· ${status.step} / ${status.total_steps}`}
      </p>
      {status.error && <p role="alert">{status.error}</p>}
      {status.q_inlet_mlps != null && <p>
        Расход, мл/с: вход {status.q_inlet_mlps.toFixed(2)} ·
        ICA {status.q_ica_mlps.toFixed(2)} · ECA {status.q_eca_mlps.toFixed(2)}.
        {" "}Невязка массы: {status.mass_imbalance_relative == null
          ? "относительная не определена" : `${(status.mass_imbalance_relative * 100).toFixed(2)}%`}.
        {(status.inlet_reversed || status.ica_reversed || status.eca_reversed) && " Обнаружен обратный поток."}
        {(status.ica_pressure_clamped || status.eca_pressure_clamped) && " Сработало ограничение давления."}
      </p>}
      <button
        disabled={busy || post === "running" || status.state === "idle"}
        onClick={() => action(() => request(route + "/su2_postprocess", {}))}
      >
        Обработать результаты
      </button>
      {post === "running" && <p role="status">Обработка результатов…</p>}
      {result && (
        <>
          {(() => {
            // Checks of the run, folded into one line; details on demand.
            const tq = result.temporal_quality, fb = result.flow_balance;
            const periodic = tq ? ({
              within_tolerance: "циклы повторяются",
              not_periodic: "циклы не совпадают",
              insufficient_cycles: "один полный цикл",
              sparse_sampling: "редкие кадры",
              insufficient_signal: "слабый сигнал",
            } as Record<string, string>)[tq.status] : null;
            const imbalance = fb?.status === "measured" && fb.relative_mean_absolute_imbalance != null
              ? `невязка потока ${(fb.relative_mean_absolute_imbalance * 100).toFixed(1)}%` : null;
            return (
              <details className="cfd-checks">
                <summary>Проверка расчёта: {[periodic, imbalance].filter(Boolean).join(" · ") || "нет данных"}</summary>
                {tq && <p>Усреднение {tq.window_start_s.toFixed(3)}–{tq.window_end_s.toFixed(3)} с
                  {tq.comparison && ["wss", "pressure"].map((name) => {
                    const m = tq.comparison[name];
                    return ` · ${name === "wss" ? "WSS" : "давление"}: RMS-различие циклов ${
                      m.relative_rms == null ? "—" : `${(m.relative_rms * 100).toFixed(1)}%`}`;
                  }).join("")}</p>}
                {fb?.status === "measured" && <p>Объём за цикл: вход {fb.inlet_volume_ml.toFixed(2)} мл ·
                  выходы {fb.outlet_volume_ml.toFixed(2)} мл · обратный поток: вход {fb.reverse_samples.inlet},
                  ICA {fb.reverse_samples.ica}, ECA {fb.reverse_samples.eca}</p>}
              </details>
            );
          })()}
          <div className="result-controls">
            <label>Показатель
              <select aria-label="Показатель CFD" value={field} onChange={(e) => setField(e.target.value)}>
                {Object.entries(FIELDS).map(([k, f]) => <option key={k} value={k}>{f.label}</option>)}
              </select>
            </label>
            <label>Шкала от
              <input type="number" step="any" value={colourRange[0]}
                onChange={(e) => setManualRange([+e.target.value, colourRange[1]])} />
            </label>
            <label>до
              <input type="number" step="any" value={colourRange[1]}
                onChange={(e) => setManualRange([colourRange[0], +e.target.value])} />
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={showStreamlines}
                onChange={(e) => setShowStreamlines(e.target.checked)} /> Линии тока (пик систолы)
            </label>
          </div>
          {timeResolved && frameIndices.length > 0 && (() => {
            const k = Math.max(0, frameIndices.indexOf(frame));
            const t0 = result.times_s[frameIndices[0]];
            const cycle = params?.cardiac_cycle_s;
            const t = result.times_s[frame] - (wholeRun ? 0 : t0);
            return (
              <div className="result-playback" aria-label="Проигрывание кадров">
                <button className="primary" disabled={!frameValues} onClick={() => setPlaying((p) => !p)}>
                  {playing ? "❚❚ Пауза" : "▶ Пуск"}
                </button>
                <button disabled={!frameValues} title="Предыдущий кадр"
                  onClick={() => { setPlaying(false); setFrame(frameIndices[(k - 1 + frameIndices.length) % frameIndices.length]); }}>‹</button>
                <button disabled={!frameValues} title="Следующий кадр"
                  onClick={() => { setPlaying(false); setFrame(frameIndices[(k + 1) % frameIndices.length]); }}>›</button>
                <input type="range" min="0" max={frameIndices.length - 1} value={k} aria-label="Кадр"
                  onChange={(e) => { setPlaying(false); setFrame(frameIndices[+e.target.value]); }} />
                <span className="time">
                  {frameValues ? <>Кадр {k + 1}/{frameIndices.length} · t = {t.toFixed(3)} с
                    {!wholeRun && cycle ? ` (${Math.round((100 * t) / cycle)}% цикла)` : ""}</> : "Загрузка кадров…"}
                </span>
                <label>Скорость
                  <select value={speed} onChange={(e) => setSpeed(+e.target.value)} aria-label="Скорость проигрывания">
                    {[0.05, 0.1, 0.25, 0.5, 1].map((v) => (
                      <option key={v} value={v}>{v === 1 ? "реальное время" : `×${v} (замедленно)`}</option>
                    ))}
                  </select>
                </label>
                <label className="checkbox">
                  <input type="checkbox" checked={wholeRun} onChange={(e) => setWholeRun(e.target.checked)} />
                  Весь расчёт (не только последний цикл)
                </label>
              </div>
            );
          })()}
          {values.length > 0 && (
            <ResultMesh
              points={result.points_mm}
              triangles={result.triangles}
              values={values}
              range={colourRange}
              title={FIELDS[field]?.label ?? field}
              streamlines={showStreamlines ? streamlines : null}
              streamRange={streamRange}
              fileName={timeResolved ? `cfd-${field}-frame${frame + 1}` : `cfd-${field}`}
              caption={timeResolved && frameValues
                ? `t = ${(result.times_s[frame] - (wholeRun ? 0 : result.times_s[frameIndices[0]] ?? 0)).toFixed(3)} с`
                : undefined}
            />
          )}
          <a href={"/api" + route + "/su2_postprocess/result"} download="cfd-results.json">
            Скачать результаты (JSON)
          </a>
        </>
      )}
    </section>
  );
}
