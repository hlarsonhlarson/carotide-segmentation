import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import { Segmentation, request } from "./Segmentation";
import { Surface } from "./Surface";
import { CFD } from "./CFD";
import { Workflow } from "./Workflow";
import { FolderPicker } from "./FolderPicker";
import { PathEditor } from "./PathEditor";
import {
  segmentationKey,
  type SegmentationPoint,
  type SegmentationChange,
} from "./segmentationGeometry";

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
  const [loader, setLoader] = useState<"dicom" | "projects" | null>("dicom");
  const [step, setStep] = useState(0);
  const [segmentations, setSegmentations] = useState<
    Record<string, SegmentationPoint[]>
  >({});
  const [autoseed, setAutoseed] = useState<any>(null);
  const [pathRevision, setPathRevision] = useState(0);
  const [reviewTarget, setReviewTarget] = useState<{
    side: string;
    branch: string;
    index: number;
  } | null>(null);
  const appliedAutoseed = useRef<string | null>(null);
  const updateSegmentation = useCallback<SegmentationChange>(
    (base, side, branch, points) => {
      const key = segmentationKey(base, side, branch);
      setSegmentations((previous) =>
        previous[key] === points ? previous : { ...previous, [key]: points },
      );
    },
    [],
  );
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
  const findSeries = useCallback(
    async (folder = path) => {
      const data = await request("/series?path=" + encodeURIComponent(folder));
      setSeries(data.series);
      setMessage(
        data.series.length
          ? `Найдено серий: ${data.series.length}. Выберите нужную ниже.`
          : "В выбранной папке DICOM-серий не найдено.",
      );
    },
    [path],
  );
  function opened(value: any) {
    setSegmentations({});
    setAutoseed(null);
    setPathRevision(0);
    setReviewTarget(null);
    appliedAutoseed.current = null;
    setStudy(value);
    setStep(0);
    setProject(value.project_dir || `/data/projects/study-${value.case_id}`);
    setMessage(value.warnings?.join("\n") || "Исследование открыто");
  }
  const base = study ? "/cases/" + study.case_id : "";
  useEffect(() => {
    if (!study) return;
    let active = true;
    async function poll() {
      try {
        const next = await request(base + "/autoseed");
        if (!active) return;
        setAutoseed(next);
        if (next.state === "done" && appliedAutoseed.current !== next.id) {
          appliedAutoseed.current = next.id;
          setPathRevision((value) => value + 1);
          setMessage(next.message || "Автоматический черновик путей готов. Проверьте его перед продолжением.");
        }
      } catch {
        // Automatic seeding is optional; do not hide the regular editor if a
        // backend without this capability is opened during an upgrade.
      }
    }
    poll();
    const timer = setInterval(poll, 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [base, study]);
  function startAutoseed() {
    run(async () => {
      const job = await request(base + "/autoseed", {});
      setAutoseed(job);
      setMessage("Запущена автоматическая сегментация. Это может занять несколько минут.");
    });
  }
  const reviewIssues = Object.entries(autoseed?.contour_qa || {}).flatMap(
    ([key, report]: [string, any]) => {
      const [issueSide, issueBranch] = key.split(":");
      return (report.rejected_sections || []).map((issue: any) => ({
        side: issueSide,
        branch: issueBranch,
        index: issue.index,
        reason: issue.reason,
      }));
    },
  );
  const autoseedStatus =
    autoseed?.stage === "preparing"
      ? "подготовка объёма"
      : autoseed?.stage === "segmenting"
        ? "модель обрабатывает КТ на CPU — это может занять несколько минут"
        : autoseed?.stage === "extracting_paths"
          ? "извлечение путей сосудов"
          : autoseed?.stage === "segmenting_contours"
            ? "создание черновых контуров"
            : "ожидание";
  function reviewIssue(issue: { side: string; branch: string; index: number }) {
    setSide(issue.side);
    setBranch(issue.branch);
    setReviewTarget(issue);
    setStep(1);
  }
  function acceptAutoseedReview() {
    run(async () => {
      const job = await request(base + "/autoseed/accept", {});
      setAutoseed(job);
      setMessage("Проверка автоматического черновика подтверждена. Результат всё ещё имеет статус unvalidated.");
    });
  }
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
          <p>Выберите папку с КТ-снимками на компьютере, затем выберите серию.</p>
          <div className="button-row centered">
            <button onClick={() => setLoader("dicom")}>
              Показать папки DICOM
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
                onSelect={
                  loader === "dicom"
                    ? (folder) =>
                        run(async () => {
                          setPath(folder);
                          await findSeries(folder);
                        })
                    : setProject
                }
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
                    onClick={() => run(() => findSeries())}
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
            <button
              disabled={busy || autoseed?.state === "queued" || autoseed?.state === "running"}
              onClick={startAutoseed}
              title="Первый запуск модели на CPU для большого КТ-объёма может занять десятки минут; расчёт продолжается, пока отображается этот статус."
            >
              {autoseed?.state === "queued" || autoseed?.state === "running"
                ? `Автосегментация: ${autoseed.progress || 0}%`
                : "Автосегментация"}
            </button>
          </div>
          {(autoseed?.state === "queued" || autoseed?.state === "running") && (
            <p className="autoseed-status" role="status">
              {autoseedStatus}. Индикатор обновится после завершения этапа модели.
            </p>
          )}
          {autoseed?.review_required && autoseed.state === "done" && (
            <section className="warning" role="status">
              <strong>Автоматический результат — черновик.</strong> Проверьте пути и контуры до построения поверхности или CFD.
              {reviewIssues.length > 0 && (
                <div className="review-list">
                  <span>Требуют исправления: {reviewIssues.length}</span>
                  {reviewIssues.slice(0, 20).map((issue, order) => (
                    <button key={`${issue.side}:${issue.branch}:${issue.index}:${order}`} onClick={() => reviewIssue(issue)}>
                      {issue.side === "right" ? "Правая" : "Левая"} · {issue.branch === "cca_ica" ? "CCA→ICA" : "CCA→ECA"} · сечение {issue.index + 1}
                    </button>
                  ))}
                  {reviewIssues.length > 20 && <small>Показаны первые 20 проблемных сечений.</small>}
                </div>
              )}
              {!autoseed.review_accepted ? (
                <button className="review-accept" disabled={busy} onClick={acceptAutoseedReview}>
                  Подтвердить проверку черновика
                </button>
              ) : (
                <p className="review-confirmed">Проверка оператором подтверждена. Статус клинической валидации не присвоен.</p>
              )}
            </section>
          )}
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
          <PathEditor
            key={base + side + pathRevision}
            base={base}
            study={study}
            side={side}
            branch={branch}
            segmentations={segmentations}
            onSeriesChange={updateSegmentation}
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
                  focusIndex={reviewTarget?.side === side && reviewTarget?.branch === branch ? reviewTarget.index : null}
                  onSeriesChange={updateSegmentation}
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
