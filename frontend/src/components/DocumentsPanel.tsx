import { ChangeEvent, Dispatch, SetStateAction, useEffect, useRef, useState } from "react";
import { api, describeError } from "../api";
import type { DocumentInfo } from "../types";
import { CameraIcon, CloseIcon, PaperclipIcon } from "./icons";

interface Props {
  sectionId: string;
  /** All of the chat's documents; library chapters are chosen with the class/subject picker and not listed here. */
  docs: DocumentInfo[];
  onDocsChange: Dispatch<SetStateAction<DocumentInfo[]>>;
  onClose: () => void;
}

export const ACCEPT = ".pdf,.docx,.txt,.md,.png,.jpg,.jpeg";
// File types whose figures are extracted (text files have none to miss).
const HAS_PICTURES = /\.(pdf|docx|png|jpe?g)$/i;
const touchScreen = typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches;

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

/** Upload files (or photos) to a chat, and report progress. Shared by "My files" and the new-chat screen. */
export function useUploads(sectionId: string, onDocsChange: Dispatch<SetStateAction<DocumentInfo[]>>) {
  const [inFlight, setInFlight] = useState<InFlight[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function uploadOne(file: File) {
    const key = `${file.name}-${file.size}-${Math.random()}`;
    setInFlight((list) => [...list, { key, name: file.name, progress: 0 }]);
    try {
      const doc = await api.uploadDocument(file, sectionId, (progress) =>
        setInFlight((list) => list.map((u) => (u.key === key ? { ...u, progress } : u))),
      );
      // The same file again: the server reused the existing document instead of a duplicate.
      if (doc.duplicate === "already_in_chat") setNotice(`${file.name} is already in this chat.`);
      else if (doc.duplicate === "existing") setNotice(`${file.name} was uploaded before, so it was added right away.`);
      onDocsChange((current) => [...current.filter((d) => d.id !== doc.id), doc]);
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
    setNotice(null);
    files.forEach(uploadOne); // in parallel; the server processes them one at a time
  }

  return { inFlight, error, notice, upload };
}

/** Buttons that open the file picker, and on phones the camera. */
export function UploadButtons({ onUpload, compact }: { onUpload: (e: ChangeEvent<HTMLInputElement>) => void; compact?: boolean }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  return (
    <div className={`upload-buttons ${compact ? "compact" : ""}`}>
      <input ref={fileRef} type="file" accept={ACCEPT} multiple hidden onChange={onUpload} />
      <input ref={cameraRef} type="file" accept="image/*" capture="environment" hidden onChange={onUpload} />
      <button className="primary" onClick={() => fileRef.current?.click()}>
        <PaperclipIcon /> Upload a file
      </button>
      {touchScreen && (
        <button className="secondary" onClick={() => cameraRef.current?.click()}>
          <CameraIcon /> Take a photo
        </button>
      )}
    </div>
  );
}

export function UploadProgress({ inFlight }: { inFlight: InFlight[] }) {
  return (
    <>
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
    </>
  );
}

/** "My files": the student's own uploads in this chat (photos of notes, PDFs, worksheets). */
export default function DocumentsPanel({ sectionId, docs, onDocsChange, onClose }: Props) {
  const [previous, setPrevious] = useState<DocumentInfo[]>([]);
  const { inFlight, error, notice, upload } = useUploads(sectionId, onDocsChange);
  const own = docs.filter((d) => !d.library);

  useEffect(() => {
    api.listDocuments().then(setPrevious).catch(() => {});
  }, [own.length]);

  async function attach(documentId: string) {
    try {
      const doc = await api.attachDocument(sectionId, documentId);
      onDocsChange((current) => [...current.filter((d) => d.id !== doc.id), doc]);
    } catch {
      /* the list refreshes on the next change */
    }
  }

  async function detach(documentId: string) {
    await api.detachDocument(sectionId, documentId);
    onDocsChange((current) => current.filter((d) => d.id !== documentId));
  }

  const attachedIds = new Set(docs.map((d) => d.id));
  const available = previous.filter((d) => !attachedIds.has(d.id) && d.status === "ready");

  return (
    <aside className="docs-panel">
      <div className="panel-header">
        <h3>My files</h3>
        <button className="icon" onClick={onClose} aria-label="Close my files">
          <CloseIcon />
        </button>
      </div>
      <p className="small muted">Your own notes, worksheets or photos. Answers use them together with the chosen subject.</p>

      <UploadButtons onUpload={upload} />
      <div className="small muted">PDF, Word, text or photos (PNG, JPG) · up to 25 MB</div>
      {error && <div className="error">{error}</div>}
      {notice && (
        <div className="small muted" role="status">
          {notice}
        </div>
      )}

      <ul className="doc-list">
        {own.length === 0 && inFlight.length === 0 && <li className="muted small empty-files">No files of your own yet.</li>}
        <UploadProgress inFlight={inFlight} />
        {own.map((d) => {
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
                <span className={`status ${d.status}`}>{working ? "reading" : d.status}</span>
                <span className="small muted">
                  {working ? "Reading… (large files take a minute)" : formatSize(d.size_bytes)}
                  {d.status === "ready" && d.figure_status === "done" && d.figure_count > 0
                    ? ` · ${d.figure_count} figure${d.figure_count === 1 ? "" : "s"}`
                    : ""}
                </span>
                {d.status === "ready" && d.figure_status === "processing" && (
                  <span className="small muted busy-label">
                    <span className="spinner" aria-hidden="true" /> finding figures…
                  </span>
                )}
                {d.status === "ready" && (d.figure_status === null || d.figure_status === "failed") && HAS_PICTURES.test(d.filename) && (
                  <span className="small muted" title="Uploaded before figure extraction, or extraction failed">
                    · figures not extracted
                  </span>
                )}
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
              <li key={d.id} className="shelf-row">
                <div className="doc-name ellipsis" title={d.filename}>
                  {d.filename}
                </div>
                <button className="secondary small" onClick={() => attach(d.id)}>
                  Add
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  );
}
