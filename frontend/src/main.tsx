import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import { Segmentation, request } from "./Segmentation";
import { Surface } from "./Surface";
import { CFD } from "./CFD";
import { Workflow } from "./Workflow";
import { FolderPicker } from "./FolderPicker";
import { PathEditor } from "./PathEditor";

function App() {
  const [study, setStudy] = useState<any>(null);
  const [path, setPath] = useState("/data/dicom");
  const [series, setSeries] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [side, setSide] = useState("right");
  const [branch, setBranch] = useState("cca_ica");
  const [project, setProject] = useState("/data/projects/study");
  const [message, setMessage] = useState("");
  const [loader, setLoader] = useState<"dicom" | "projects" | null>(null);
  const [step, setStep] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (step) dialog.current?.showModal();
    else dialog.current?.close();
  }, [step]);
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
    setProject(value.project_dir || `/data/projects/study-${value.case_id}`);
    setMessage(value.warnings?.join("\n") || "Исследование открыто");
  }
  const base = study ? "/cases/" + study.case_id : "";
  async function save() {
    await request(base + "/project", { path: project });
    await request(base + "/save/project", {});
    setMessage("Проект сохранён. Автосохранение включено.");
  }
  const titles = [
    "",
    "Контуры сосуда",
    "Поверхность",
    "Расчётная сетка",
    "Граничные условия",
    "CFD Computation",
  ];
  return (
    <main>
      <header className="app-toolbar">
        <span>Carotid Segmentation</span>
        {study && (
          <button
            onClick={() => {
              setStudy(null);
              setMessage("");
              setError("");
            }}
          >
            Открыть другое исследование…
          </button>
        )}
      </header>
      {!study ? (
        <div className="case-loader">
          <h1>Carotid Segmentation</h1>
          <p>
            Откройте папку DICOM или продолжите работу с сохранённым проектом.
          </p>
          <div className="button-row centered">
            <button className="primary" onClick={() => setLoader("dicom")}>
              Выбрать папку DICOM…
            </button>
            <button onClick={() => setLoader("projects")}>
              Открыть сохранённый проект…
            </button>
          </div>
          {loader && (
            <section className="loader-panel panel">
              <FolderPicker
                key={loader}
                kind={loader}
                onSelect={loader === "dicom" ? setPath : setProject}
              />
              {loader === "dicom" ? (
                <>
                  <label>
                    Папка снимков
                    <input
                      value={path}
                      onChange={(e) => setPath(e.target.value)}
                    />
                  </label>
                  <button
                    disabled={busy}
                    className="primary"
                    onClick={() =>
                      run(async () => {
                        const data = await request(
                          "/series?path=" + encodeURIComponent(path),
                        );
                        setSeries(data.series);
                        if (!data.series.length)
                          setMessage(
                            "В выбранной папке DICOM-серий не найдено.",
                          );
                      })
                    }
                  >
                    Найти DICOM-серии
                  </button>
                  <div className="series-list">
                    {series.map((s) => (
                      <button
                        className="series-item"
                        key={s.series_uid}
                        disabled={busy}
                        onClick={() =>
                          run(async () =>
                            opened(
                              await request("/cases", {
                                path,
                                series_uid: s.series_uid,
                              }),
                            ),
                          )
                        }
                      >
                        <strong>{s.description || "Серия без описания"}</strong>
                        <small>
                          {s.modality} · {s.n_files} файлов
                        </small>
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <label>
                    Папка проекта
                    <input
                      value={project}
                      onChange={(e) => setProject(e.target.value)}
                    />
                  </label>
                  <button
                    className="primary"
                    disabled={busy}
                    onClick={() =>
                      run(async () =>
                        opened(
                          await request("/projects/open", { path: project }),
                        ),
                      )
                    }
                  >
                    Открыть проект
                  </button>
                </>
              )}
            </section>
          )}
          {busy && <p role="status">Загрузка…</p>}
          {message && <p role="status">{message}</p>}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </div>
      ) : (
        <>
          <div className="study-meta">
            <span>{study.modality}</span>
            <span>
              {study.spacing_mm.map((n: number) => n.toFixed(2)).join(" × ")} мм
            </span>
            <span>{study.shape.join(" × ")} vox</span>
            <span className="study-status" role="status">
              {message}
            </span>
          </div>
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
          <PathEditor
            key={base + side}
            base={base}
            study={study}
            side={side}
            branch={branch}
            onSide={setSide}
            onBranch={setBranch}
            onStep={setStep}
            onError={setError}
            workflow={
              <>
                <Workflow
                  base={base}
                  side={side}
                  currentStep={step}
                  onStep={setStep}
                  onSave={() => run(save)}
                  busy={busy}
                />
                <details className="panel project-panel">
                  <summary>Проект</summary>
                  <FolderPicker kind="projects" onSelect={setProject} />
                  <label>
                    Папка проекта
                    <input
                      value={project}
                      onChange={(e) => setProject(e.target.value)}
                    />
                  </label>
                  <button disabled={busy} onClick={() => run(save)}>
                    Сохранить проект
                  </button>
                </details>
              </>
            }
          />
          <dialog
            className={step === 1 ? "segmentation-dialog" : undefined}
            ref={dialog}
            onCancel={() => setStep(0)}
            onClose={() => setStep(0)}
          >
            <div className="dialog-header">
              <strong>
                {titles[step]} · {side === "right" ? "Правая" : "Левая"}
              </strong>
              <button aria-label="Закрыть окно" onClick={() => setStep(0)}>
                Готово
              </button>
            </div>
            {error && (
              <p role="alert" className="error">
                {error}
              </p>
            )}
            <div className="dialog-content">
              {step === 1 && (
                <Segmentation
                  key={base + side + branch}
                  base={base}
                  side={side}
                  branch={branch}
                  study={study}
                  onError={setError}
                />
              )}
              {(step === 2 || step === 3) && (
                <Surface
                  key={base + side + step}
                  base={base}
                  side={side}
                  initialKind={step === 3 ? "mesh" : "surface"}
                  onError={setError}
                />
              )}
              {(step === 4 || step === 5) && (
                <CFD
                  key={base + side}
                  base={base}
                  side={side}
                  onError={setError}
                />
              )}
            </div>
          </dialog>
        </>
      )}
    </main>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
