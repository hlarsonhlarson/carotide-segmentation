import { useEffect, useState } from "react";
import { ResultMesh } from "./ResultMesh";
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
    [frameValues, setFrameValues] = useState<number[] | null>(null);
  useEffect(() => {
    let active = true;
    setFrameValues(null);
    if (result && (field === "wss" || field === "pressure"))
      request(route + "/su2_postprocess/frame/" + frame + "?field=" + field)
        .then((d) => {
          if (active) setFrameValues(d.values);
        })
        .catch((e) => {
          if (active) onError(String(e));
        });
    return () => {
      active = false;
    };
  }, [route, result, field, frame]);
  const values: number[] = result
    ? field === "wss" || field === "pressure"
      ? frameValues || []
      : result[field]
    : [];
  useEffect(() => {
    let active = true;
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
      <button
        disabled={busy || post === "running" || status.state === "idle"}
        onClick={() => action(() => request(route + "/su2_postprocess", {}))}
      >
        Обработать результаты
      </button>
      <p>Обработка: {post}</p>
      <button
        disabled={busy || post !== "done"}
        onClick={() =>
          action(async () =>
            setResult(await request(route + "/su2_postprocess/result")),
          )
        }
      >
        Открыть результаты
      </button>
      {result && (
        <>
          <p>
            Вычисление завершено. Физическая достоверность ещё не подтверждена.
          </p>
          <select
            aria-label="Показатель CFD"
            value={field}
            onChange={(e) => setField(e.target.value)}
          >
            {[
              ["tawss_pa", "TAWSS, Па"],
              ["osi", "OSI"],
              ["rrt_pa_inv", "RRT, 1/Па"],
              ["mean_pressure_pa", "Среднее давление, Па"],
              ["wss", "WSS по времени, Па"],
              ["pressure", "Давление по времени, Па"],
            ].map(([k, t]) => (
              <option key={k} value={k}>
                {t}
              </option>
            ))}
          </select>
          {(field === "wss" || field === "pressure") && (
            <label>
              Кадр {frame + 1} · {result.times_s[frame]} с
              <input
                type="range"
                min="0"
                max={result.times_s.length - 1}
                value={frame}
                onChange={(e) => setFrame(+e.target.value)}
              />
            </label>
          )}
          {values.length > 0 && (
            <ResultMesh
              points={result.points_mm}
              triangles={result.triangles}
              values={values}
            />
          )}
          <p>
            Узлов: {result.points_mm.length} · Кадров: {result.times_s.length}
          </p>
          <p>
            Диапазон:{" "}
            {values
              .reduce((a: number, b: number) => Math.min(a, b), Infinity)
              .toPrecision(4)}{" "}
            —{" "}
            {values
              .reduce((a: number, b: number) => Math.max(a, b), -Infinity)
              .toPrecision(4)}
          </p>
          <a
            href={"/api" + route + "/su2_postprocess/result"}
            download="cfd-results.json"
          >
            Скачать результаты JSON
          </a>
          <button
            disabled={busy}
            onClick={() =>
              action(() => request(base + "/save/cfd_results/" + side, {}))
            }
          >
            Сохранить результаты в проект
          </button>
        </>
      )}
    </section>
  );
}
