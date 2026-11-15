import { useEffect, useMemo, useRef, useState } from "react";
import { request } from "./Segmentation";

type Folder = { name: string; path: string; project: boolean; files: number | null };
type Listing = { path: string; absolute_path: string; folders: Folder[] };

// A plain file-browser list: breadcrumbs, one folder per row, click to
// select, double-click (or Enter) to open, arrows to move.
export function FolderPicker({
  kind,
  onSelect,
}: {
  kind: "dicom" | "projects";
  onSelect: (path: string) => void;
}) {
  const [path, setPath] = useState("");
  const [data, setData] = useState<Listing | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [newName, setNewName] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    request("/storage?kind=" + kind + "&path=" + encodeURIComponent(path))
      .then((d) => {
        if (!active) return;
        setData(d);
        setError("");
      })
      .catch((e) => active && setError(String(e)));
    return () => {
      active = false;
    };
  }, [path, kind, reload]);

  const folders = useMemo(
    () => (data?.folders || []).filter((f) => f.name.toLowerCase().includes(filter.trim().toLowerCase())),
    [data, filter],
  );
  const crumbs = path ? path.split("/") : [];
  const target = folders.find((f) => f.path === selected);
  const targetAbsolute = target && data
    ? `${data.absolute_path}/${target.name}`
    : data?.absolute_path ?? "";

  function open(next: string) {
    setPath(next);
    setSelected(null);
    setFilter("");
  }
  function up() {
    if (path) open(crumbs.slice(0, -1).join("/"));
  }
  function choose(folder?: Folder) {
    if (!data) return;
    onSelect(folder ? `${data.absolute_path}/${folder.name}` : targetAbsolute);
  }
  function activate(folder: Folder) {
    // A saved project is a destination, not something to browse into.
    if (kind === "projects" && folder.project) choose(folder);
    else open(folder.path);
  }
  function onKey(e: React.KeyboardEvent) {
    if (!folders.length) return;
    const index = folders.findIndex((f) => f.path === selected);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const next = e.key === "ArrowDown" ? Math.min(folders.length - 1, index + 1) : Math.max(0, index - 1);
      setSelected(folders[next].path);
      list.current?.querySelectorAll<HTMLElement>("[role=option]")[next]?.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && target) {
      e.preventDefault();
      activate(target);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      up();
    }
  }
  async function createFolder() {
    if (!newName?.trim()) return;
    try {
      const made = await request("/storage/folder", { path, name: newName.trim() });
      setNewName(null);
      setReload((r) => r + 1);
      setSelected(made.path);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <section className="folder-browser" aria-label="Выбор папки">
      <div className="folder-browser-title">
        <strong>{kind === "dicom" ? "Папка со снимками" : "Папка проекта"}</strong>
        {kind === "projects" && <span>Проекту нужна своя пустая папка: создайте её кнопкой «Новая папка».</span>}
      </div>
      <nav className="folder-browser-crumbs" aria-label="Путь">
        <button onClick={() => open("")}>{kind === "dicom" ? "Снимки" : "Проекты"}</button>
        {crumbs.map((name, i) => (
          <span key={i}>
            <span className="sep">›</span>
            <button onClick={() => open(crumbs.slice(0, i + 1).join("/"))}>{name}</button>
          </span>
        ))}
      </nav>
      <div className="folder-browser-toolbar">
        <button disabled={!path} onClick={up} title="Вверх (Backspace)">↑ Вверх</button>
        <input
          type="search"
          placeholder="Поиск по имени"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Поиск папки"
        />
        {kind === "projects" && (newName === null
          ? <button onClick={() => setNewName("")}>Новая папка</button>
          : <span className="folder-browser-new">
              <input autoFocus value={newName} placeholder="Имя папки"
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") createFolder(); if (e.key === "Escape") setNewName(null); }} />
              <button className="primary" onClick={createFolder}>Создать</button>
              <button onClick={() => setNewName(null)}>Отмена</button>
            </span>)}
      </div>
      {error && <p role="alert">{error}</p>}
      <div className="folder-browser-list" role="listbox" tabIndex={0} ref={list} onKeyDown={onKey}
        aria-label="Папки">
        {data && !folders.length && (
          <div className="folder-browser-empty">
            {filter ? "Ничего не найдено" : "Вложенных папок нет"}
          </div>
        )}
        {folders.map((f) => (
          <div
            key={f.path}
            role="option"
            aria-selected={selected === f.path}
            className="folder-browser-row"
            onClick={() => setSelected(f.path)}
            onDoubleClick={() => activate(f)}
            title={kind === "projects" && f.project ? "Двойной щелчок — выбрать проект" : "Двойной щелчок — открыть"}
          >
            <span className="icon">{f.project ? "🗂" : "📁"}</span>
            <span className="name">{f.name}</span>
            <span className="meta">
              {f.project ? "проект" : f.files ? `${f.files} файлов` : ""}
            </span>
          </div>
        ))}
      </div>
      <div className="folder-browser-footer">
        <span className="target" title={targetAbsolute}>{targetAbsolute}</span>
        <button className="primary" disabled={!data} onClick={() => choose(target)}>
          {kind === "dicom" ? "Открыть эту папку" : "Выбрать"}
        </button>
      </div>
    </section>
  );
}
