import { useEffect, useState } from "react";
type Point = { contour_px?: number[][]; radius_mm?: number; method?: string };
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
  onError,
}: {
  base: string;
  side: string;
  branch: string;
  onError: (message: string) => void;
}) {
  const [points, setPoints] = useState<Point[]>([]),
    [index, setIndex] = useState(0),
    [method, setMethod] = useState("auto"),
    [radius, setRadius] = useState(3),
    [manual, setManual] = useState<number[][]>([]),
    [busy, setBusy] = useState(false);
  const route = `${base}/segment/${side}/${branch}`;
  async function refresh() {
    const data = await request(route);
    setPoints(data.points);
    setIndex((i) => Math.min(i, Math.max(0, data.points.length - 1)));
  }
  useEffect(() => {
    let active = true;
    setPoints([]);
    setIndex(0);
    setManual([]);
    request(route)
      .then((d) => {
        if (active) setPoints(d.points);
      })
      .catch((e) => {
        if (active) onError(String(e));
      });
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
  return (
    <section>
      <h2>Контуры сосуда</h2>
      <button
        disabled={busy}
        onClick={() => action(() => request(route + "/prepare", {}))}
      >
        Подготовить сечения заново
      </button>
      {points.length > 0 && (
        <>
          <label>
            Сечение {index + 1} / {points.length}
            <input
              type="range"
              min={0}
              max={points.length - 1}
              value={index}
              onChange={(e) => {
                setIndex(+e.target.value);
                setManual([]);
              }}
            />
          </label>
          <div
            style={{
              position: "relative",
              width: "min(100%, 480px)",
              aspectRatio: "1",
            }}
          >
            <img
              style={{ width: "100%", height: "100%" }}
              src={"/api" + route + "/slice/" + index}
              alt="Поперечное сечение сосуда"
            />
            <svg
              viewBox="0 0 160 160"
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                cursor: method === "manual" ? "crosshair" : "default",
              }}
              onClick={(e) => {
                if (method !== "manual" || busy) return;
                const r = e.currentTarget.getBoundingClientRect();
                setManual((p) => [
                  ...p,
                  [
                    ((e.clientY - r.top) / r.height) * 160,
                    ((e.clientX - r.left) / r.width) * 160,
                  ],
                ]);
              }}
            >
              <polygon
                points={(method === "manual"
                  ? manual
                  : (points[index]?.contour_px ?? [])
                )
                  .map((p) => `${p[1]},${p[0]}`)
                  .join(" ")}
                fill="#40ddc522"
                stroke="#40ddc5"
                strokeWidth=".7"
              />
            </svg>
          </div>
          <label>
            Метод
            <select
              value={method}
              onChange={(e) => {
                setMethod(e.target.value);
                setManual([]);
              }}
            >
              {[
                ["auto", "Автоматически"],
                ["threshold", "Порог"],
                ["levelset", "Активный контур"],
                ["circle", "Круг"],
                ["manual", "Ручной контур"],
              ].map(([v, t]) => (
                <option value={v} key={v}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label>
            Начальный радиус, мм
            <input
              type="number"
              min="0.1"
              max="50"
              step="0.1"
              value={radius}
              onChange={(e) => setRadius(+e.target.value)}
            />
          </label>
          <button
            disabled={busy || (method === "manual" && manual.length < 3)}
            onClick={() =>
              action(() =>
                request(
                  route +
                    `/point/${index}` +
                    (method === "manual" ? "/manual" : ""),
                  method === "manual"
                    ? { contour_px: manual }
                    : { method, radius_mm: radius },
                ),
              )
            }
          >
            Применить к сечению
          </button>
          <button
            disabled={busy || method === "manual"}
            onClick={() =>
              action(async () => {
                for (let i = 0; i < points.length; i++)
                  await request(route + `/point/${i}`, {
                    method,
                    radius_mm: radius,
                  });
              })
            }
          >
            Обработать все сечения
          </button>
          <button
            disabled={busy}
            onClick={() =>
              action(() => request(route + `/point/${index}`, {}, "DELETE"))
            }
          >
            Удалить контур
          </button>
          <p>
            {points.filter((p) => p.contour_px?.length).length} из{" "}
            {points.length} контуров ·{" "}
            {busy ? "Обработка…" : "Изменения сохраняются в выбранный проект"}
          </p>
        </>
      )}
    </section>
  );
}
