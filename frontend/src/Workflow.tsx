import { useEffect, useState } from "react";
import { request } from "./Segmentation";
const stages = [
  ["pathway", "Pathway", "Пути CCA → ICA и CCA → ECA"],
  ["segmentation", "Segmentation", "Контуры стенки сосуда"],
  ["surface", "Surface", "Построение 3D-поверхности"],
  ["meshing", "Meshing", "Сетка для расчёта кровотока"],
  ["boundaryConditions", "Boundary Conditions", "Параметры входа и выходов"],
  ["cfdComputation", "CFD Computation", "Пульсирующий кровоток"],
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
export function Workflow({
  base,
  side,
  currentStep,
  onStep,
  onSave,
  busy,
}: {
  base: string;
  side: string;
  currentStep: number;
  onStep: (step: number) => void;
  onSave: () => void;
  busy: boolean;
}) {
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
    const timer = setInterval(update, 2000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [base, side]);
  return (
    <nav className="workflow panel" aria-label="Этапы работы">
      <h2 className="eyebrow">WORKFLOW</h2>
      {stages.map(([key, label, description], i) => (
        <div className="workflow-row" key={key}>
          <button
            className="step-link"
            aria-current={i === currentStep ? "step" : undefined}
            onClick={() => onStep(i)}
          >
            <span
              className={`step-badge ${state[key] === "done" || i === currentStep ? "active" : ""}`}
            >
              {state[key] === "done" ? "✓" : i + 1}
            </span>
            <span>
              <strong>{label}</strong>
              <small>{description}</small>
              <small className="step-status">
                {names[state[key]] || "Подключение…"}
              </small>
            </span>
          </button>
          <button
            className="save-step"
            title="Сохранить проект со всеми этапами"
            aria-label={`Сохранить ${label}`}
            disabled={busy}
            onClick={onSave}
          >
            Save
          </button>
        </div>
      ))}
    </nav>
  );
}
