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
    <details>
      <summary>Выбрать папку</summary>
      {error && <p role="alert">{error}</p>}
      {data && (
        <>
          <p>{data.path === "." ? "Корневая папка" : data.path}</p>
          <button
            onClick={() => setPath(path.split("/").slice(0, -1).join("/"))}
          >
            На уровень выше
          </button>
          {data.folders.map((f: any) => (
            <button key={f.path} onClick={() => setPath(f.path)}>
              📁 {f.name}
            </button>
          ))}
          <button onClick={() => onSelect(data.absolute_path)}>
            Выбрать эту папку
          </button>
        </>
      )}
    </details>
  );
}
