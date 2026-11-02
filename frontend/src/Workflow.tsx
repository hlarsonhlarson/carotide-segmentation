import { useEffect, useState } from "react";
import { request } from "./Segmentation";
const stages = [
  ["pathway", "Путь"],
  ["segmentation", "Контуры"],
  ["surface", "Поверхность"],
  ["meshing", "Сетка"],
  ["boundaryConditions", "Условия"],
  ["cfdComputation", "Расчёт"],
];
const names: Record<string, string> = {
  done: "Готово",
  idle: "Не выполнено",
  partial: "Частично",
  stale: "Нужно пересчитать",
  running: "Выполняется",
  error: "Ошибка",
  cancelled: "Отменено",
};
export function Workflow({ base, side }: { base: string; side: string }) {
  const [state, setState] = useState<Record<string, string>>({});
  useEffect(() => {
    let alive = true;
    async function update() {
      try {
        const s = await request(base + "/workflow/" + side);
        if (alive) setState(s);
      } catch {
        if (alive) setState({});
      }
    }
    update();
    const timer = setInterval(update, 1500);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [base, side]);
  return (
    <nav aria-label="Этапы работы">
      {stages.map(([key, label]) => (
        <p key={key}>
          {state[key] === "done" ? "✓" : "○"} {label}{" "}
          <small>{names[state[key]] || "Нет связи"}</small>
        </p>
      ))}
    </nav>
  );
}
