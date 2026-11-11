import { useEffect, useState } from "react";
import { request } from "./Segmentation";
export function FolderPicker({
  kind,
  onSelect,
}: {
  kind: "dicom" | "projects";
  onSelect: (path: string) => void;
}) {
  const [path, setPath] = useState(""),
    [data, setData] = useState<any>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    request("/storage?kind=" + kind + "&path=" + encodeURIComponent(path))
      .then((d) => {
        if (active) {
          setData(d);
          setError("");
        }
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, [path, kind]);
  return (
    <section className="folder-picker" aria-label="Выбор папки">
      <strong>{kind === "dicom" ? "Папка со снимками" : "Папка проектов"}</strong>
      {error && <p role="alert">{error}</p>}
      {data && (
        <>
          <div className="folder-picker-path">
            {data.path === "." ? "Корневая папка" : data.path}
          </div>
          <div className="folder-picker-actions">
            <button
              disabled={!path}
              onClick={() => setPath(path.split("/").slice(0, -1).join("/"))}
            >
              ← Выше
            </button>
            <button className="primary" onClick={() => onSelect(data.absolute_path)}>
              Использовать эту папку
            </button>
          </div>
          {data.folders.length ? (
            <div className="folder-picker-folders">
              {data.folders.map((f: any) => (
                <button key={f.path} onClick={() => setPath(f.path)}>
                  📁 {f.name}
                </button>
              ))}
            </div>
          ) : (
            <p>В этой папке нет вложенных каталогов.</p>
          )}
        </>
      )}
    </section>
  );
}
