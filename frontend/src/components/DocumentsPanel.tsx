import { ChangeEvent, Dispatch, SetStateAction, useEffect, useRef, useState } from "react";
import { api, describeError } from "../api";
import type { DocumentInfo } from "../types";

interface Props {
  sectionId: string;
  docs: DocumentInfo[];
  onDocsChange: Dispatch<SetStateAction<DocumentInfo[]>>;
  onClose: () => void;
}

const ACCEPT = ".pdf,.docx,.txt,.md,.png,.jpg,.jpeg";

interface InFlight {
  key: string;
  name: string;
  progress: number; // 0..1 of the file sent to the server
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function DocumentsPanel({ sectionId, docs, onDocsChange, onClose }: Props) {
  const [library, setLibrary] = useState<DocumentInfo[]>([]);
  const [inFlight, setInFlight] = useState<InFlight[]>([]);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.listDocuments().then(setLibrary).catch(() => {});
  }, [docs.length]);

  async function uploadOne(file: File) {
    const key = `${file.name}-${file.size}-${Math.random()}`;
    setInFlight((list) => [...list, { key, name: file.name, progress: 0 }]);
    try {
      const doc = await api.uploadDocument(file, sectionId, (progress) =>
        setInFlight((list) => list.map((u) => (u.key === key ? { ...u, progress } : u))),
      );
      onDocsChange((current) => [...current, doc]);
    } catch (err) {
      setError(`${file.name}: ${describeError(err, "upload failed")}`);
    } finally {
      setInFlight((list) => list.filter((u) => u.key !== key));
    }
  }

  function upload(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    setError(null);
    files.forEach(uploadOne); // in parallel; the server processes them one at a time
  }

  async function attach(documentId: string) {
    const doc = await api.attachDocument(sectionId, documentId);
    onDocsChange((current) => [...current, doc]);
  }

  async function detach(documentId: string) {
    await api.detachDocument(sectionId, documentId);
    onDocsChange((current) => current.filter((d) => d.id !== documentId));
  }

  const attachedIds = new Set(docs.map((d) => d.id));
  const available = library.filter((d) => !attachedIds.has(d.id) && d.status === "ready");

  return (
    <aside className="docs-panel">
      <div className="panel-header">
        <h3>Documents in this chat</h3>
        <button className="icon" onClick={onClose} aria-label="Close documents">
          ×
        </button>
      </div>
      <p className="small muted">Answers and quizzes use only these files.</p>

      <input ref={inputRef} type="file" accept={ACCEPT} multiple hidden onChange={upload} />
      <button className="primary full" onClick={() => inputRef.current?.click()}>
        Upload files
      </button>
      <div className="small muted">PDF, DOCX, TXT, MD, PNG, JPG · up to 25 MB</div>
      {error && <div className="error">{error}</div>}

      <ul className="doc-list">
        {docs.length === 0 && inFlight.length === 0 && <li className="muted small">No documents yet.</li>}
        {inFlight.map((u) => (
          <li key={u.key} className="busy" aria-busy="true">
            <div className="doc-name ellipsis" title={u.name}>
              <span className="spinner" aria-hidden="true" />
              {u.name}
            </div>
            <div className="progress" role="progressbar" aria-valuenow={Math.round(u.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div className="progress-fill" style={{ width: `${Math.max(3, u.progress * 100)}%` }} />
            </div>
            <div className="small muted">
              {u.progress < 1 ? `Uploading… ${Math.round(u.progress * 100)}%` : "Upload complete, saving…"}
            </div>
          </li>
        ))}
        {docs.map((d) => {
          const working = d.status === "pending" || d.status === "processing";
          return (
            <li key={d.id} className={working ? "busy" : undefined} aria-busy={working}>
              <div className="doc-name ellipsis" title={d.filename}>
                {working && <span className="spinner" aria-hidden="true" />}
                {d.filename}
              </div>
              {working && (
                <div className="progress indeterminate" role="progressbar" aria-label="Processing">
                  <div className="progress-fill" />
                </div>
              )}
              <div className="doc-meta">
                <span className={`status ${d.status}`}>{working ? "processing" : d.status}</span>
                <span className="small muted">
                  {working ? "Reading & indexing… (large files can take a minute)" : formatSize(d.size_bytes)}
                  {d.status === "ready" ? ` · ${d.chunk_count} chunks` : ""}
                </span>
                {!working && (
                  <button className="link small" onClick={() => detach(d.id)}>
                    Remove
                  </button>
                )}
              </div>
              {d.error && <div className="error small">{d.error}</div>}
            </li>
          );
        })}
      </ul>

      {available.length > 0 && (
        <>
          <h4>From your other chats</h4>
          <ul className="doc-list">
            {available.map((d) => (
              <li key={d.id}>
                <div className="doc-name ellipsis" title={d.filename}>
                  {d.filename}
                </div>
                <button className="secondary small" onClick={() => attach(d.id)}>
                  Add to chat
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  );
}
