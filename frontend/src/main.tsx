import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import { Segmentation, request } from "./Segmentation";
import { Surface } from "./Surface";
import { CFD } from "./CFD";
import { Workflow } from "./Workflow";
import { FolderPicker } from "./FolderPicker";
import { PathEditor } from "./PathEditor";
function App() {
  const [study, setStudy] = useState<any>(null),
    [path, setPath] = useState("/data/dicom"),
    [series, setSeries] = useState<any[]>([]);
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [side, setSide] = useState("right"),
    [branch, setBranch] = useState("cca_ica");
  const [project, setProject] = useState("/data/projects/study"),
    [message, setMessage] = useState(""),
    [step, setStep] = useState(0);
  async function run(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  function opened(value: any) {
    setStudy(value);
    setStep(0);
    setMessage(value.warnings?.join("\n") || "Исследование открыто");
  }
  const base = study ? "/cases/" + study.case_id : "";
  return (
    <main>
      <header>
        <b>
          CAROTID <small>WORKBENCH</small>
        </b>
        <span>Исследование сосудов</span>
      </header>
      <section className="layout">
        <aside>
          <h1>Рабочее пространство</h1>
          <p>От снимков к модели кровотока</p>
          <FolderPicker kind="dicom" onSelect={setPath} />
          <label>
            Папка снимков
            <input value={path} onChange={(e) => setPath(e.target.value)} />
          </label>
          <button
            disabled={busy}
            onClick={() =>
              run(async () =>
                setSeries(
                  (await request("/series?path=" + encodeURIComponent(path)))
                    .series,
                ),
              )
            }
          >
            Найти DICOM-серии
          </button>
          {series.map((s) => (
            <button
              key={s.series_uid}
              disabled={busy}
              onClick={() =>
                run(async () =>
                  opened(
                    await request("/cases", { path, series_uid: s.series_uid }),
                  ),
                )
              }
            >
              {s.description || s.series_uid} · {s.n_files} файлов
            </button>
          ))}
          <FolderPicker kind="projects" onSelect={setProject} />
          <label>
            Папка проекта
            <input
              value={project}
              onChange={(e) => setProject(e.target.value)}
            />
          </label>
          <button
            disabled={busy}
            onClick={() =>
              run(async () =>
                opened(await request("/projects/open", { path: project })),
              )
            }
          >
            Открыть проект
          </button>
          {study && (
            <>
              <button
                disabled={busy}
                onClick={() =>
                  run(async () => {
                    await request(base + "/project", { path: project });
                    await request(base + "/save/project", {});
                    setMessage("Проект сохранён. Автосохранение включено.");
                  })
                }
              >
                Сохранить проект
              </button>
              <label>
                Сторона
                <select value={side} onChange={(e) => setSide(e.target.value)}>
                  <option value="right">Правая</option>
                  <option value="left">Левая</option>
                </select>
              </label>
              <label>
                Ветвь
                <select
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                >
                  <option value="cca_ica">CCA → ICA</option>
                  <option value="cca_eca">CCA → ECA</option>
                </select>
              </label>
              <Workflow key={base + side} base={base} side={side} />
            </>
          )}
        </aside>
        <article>
          {error && (
            <div role="alert" className="error">
              {error}
            </div>
          )}
          {message && <p role="status">{message}</p>}
          {study ? (
            <>
              <nav className="tabs" aria-label="Разделы исследования">
                {["Пути", "Контуры", "Геометрия", "Расчёт"].map((name, i) => (
                  <button
                    key={name}
                    aria-pressed={step === i}
                    onClick={() => {
                      setStep(i);
                      setError("");
                    }}
                  >
                    {name}
                  </button>
                ))}
              </nav>
              {step === 0 && (
                <PathEditor
                  key={base + side + branch}
                  base={base}
                  side={side}
                  branch={branch}
                  study={study}
                  onError={setError}
                />
              )}
              {step === 1 && (
                <Segmentation
                  key={base + side + branch}
                  base={base}
                  side={side}
                  branch={branch}
                  onError={setError}
                />
              )}
              {step === 2 && (
                <Surface
                  key={base + side}
                  base={base}
                  side={side}
                  onError={setError}
                />
              )}
              {step === 3 && (
                <CFD
                  key={base + side}
                  base={base}
                  side={side}
                  onError={setError}
                />
              )}
            </>
          ) : (
            <div className="empty">
              <h2>Откройте исследование</h2>
              <p>Выберите DICOM-серию или сохранённый проект слева.</p>
              <p>
                Исходные снимки подключаются только для чтения. Для каждого
                исследования используйте отдельную папку проекта.
              </p>
            </div>
          )}
        </article>
      </section>
    </main>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
