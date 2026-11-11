import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import { api, describeError } from "../api";
import { bookTitle, matches, shelves } from "../library";
import type { DocumentInfo } from "../types";

const ACCEPT = ".pdf,.docx,.txt,.md,.png,.jpg,.jpeg";
const GRADES = ["6", "7", "8", "9", "10", "11", "12"];
const SUBJECTS = ["Mathematics", "Science", "Physics", "Chemistry", "Biology", "English", "Social Science"];

interface InFlight {
  key: string;
  name: string;
  progress: number;
}

function isWorking(d: DocumentInfo): boolean {
  return d.status === "pending" || d.status === "processing" || (d.status === "ready" && d.figure_status === "processing");
}

/** Admin page: add books to the shared library, watch them being processed, remove them. */
export default function LibraryManager() {
  const [books, setBooks] = useState<DocumentInfo[]>([]);
  const [grade, setGrade] = useState("10");
  const [subject, setSubject] = useState("Mathematics");
  const [query, setQuery] = useState("");
  const [inFlight, setInFlight] = useState<InFlight[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => api.listLibrary().then(setBooks).catch(() => {}), []);
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Books are processed one at a time on the server; poll while any is still in the queue.
  const working = books.filter(isWorking).length;
  useEffect(() => {
    if (working === 0 && inFlight.length === 0) return;
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, [working, inFlight.length, refresh]);

  async function uploadOne(file: File, g: string, s: string) {
    const key = `${file.name}-${file.size}-${Math.random()}`;
    setInFlight((list) => [...list, { key, name: file.name, progress: 0 }]);
    try {
      const doc = await api.addToLibrary(file, g, s, (progress) =>
        setInFlight((list) => list.map((u) => (u.key === key ? { ...u, progress } : u))),
      );
      setBooks((list) => [...list.filter((d) => d.id !== doc.id), doc]);
    } catch (err) {
      setErrors((list) => [...list, `${file.name}: ${describeError(err, "upload failed")}`]);
    } finally {
      setInFlight((list) => list.filter((u) => u.key !== key));
    }
  }

  async function upload(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    setErrors([]);
    // A few at a time: dozens of chapters at once would compete for the connection.
    const queue = [...files];
    const g = grade.trim();
    const s = subject.trim();
    await Promise.all(
      Array.from({ length: Math.min(3, queue.length) }, async () => {
        for (let file = queue.shift(); file; file = queue.shift()) await uploadOne(file, g, s);
      }),
    );
  }

  async function reprocess(doc: DocumentInfo) {
    try {
      const updated = await api.reprocessLibraryDocument(doc.id);
      setBooks((list) => list.map((d) => (d.id === doc.id ? { ...updated, figure_status: "processing" } : d)));
    } catch (err) {
      setErrors([`${bookTitle(doc)}: ${describeError(err, "couldn't process it again")}`]);
    }
  }

  async function remove(doc: DocumentInfo) {
    if (!confirm(`Delete "${bookTitle(doc)}" from the library? It is removed from every student's chats too.`)) return;
    try {
      await api.removeFromLibrary(doc.id);
      setBooks((list) => list.filter((d) => d.id !== doc.id));
    } catch (err) {
      setErrors([describeError(err, "Couldn't delete that book.")]);
    }
  }

  const ready = books.filter((d) => d.status === "ready").length;
  const failed = books.filter((d) => d.status === "failed").length;
  const waiting = books.filter((d) => d.vision_pending || d.figure_status === "waiting").length;

  return (
    <div className="manager">
      <header className="chat-header">
        <h2>Shared library</h2>
        <span className="small muted">
          {ready} ready{working ? ` · ${working} processing` : ""}
          {failed ? ` · ${failed} failed` : ""}
          {waiting ? ` · ${waiting} waiting for AI limits` : ""}
        </span>
      </header>
      <div className="manager-body">
        <section className="manager-card">
          <h3>Add books</h3>
          <p className="small muted">
            Every student can add these to their chats and ask about them. Upload one PDF per chapter, named the way
            students should see it (e.g. “Ch 2 Polynomials.pdf”). Files are processed one at a time; math pages are
            read as LaTeX, so a chapter can take a minute or two.
          </p>
          <div className="manager-fields">
            <label>
              Class
              <input list="library-grades" value={grade} onChange={(e) => setGrade(e.target.value)} maxLength={20} />
              <datalist id="library-grades">
                {GRADES.map((g) => (
                  <option key={g} value={g} />
                ))}
              </datalist>
            </label>
            <label>
              Subject
              <input list="library-subjects" value={subject} onChange={(e) => setSubject(e.target.value)} maxLength={60} />
              <datalist id="library-subjects">
                {SUBJECTS.map((s) => (
                  <option key={s} value={s} />
                ))}
              </datalist>
            </label>
          </div>
          <input ref={inputRef} type="file" accept={ACCEPT} multiple hidden onChange={upload} />
          <button className="primary" onClick={() => inputRef.current?.click()}>
            Choose files for {[grade.trim() && `Class ${grade.trim()}`, subject.trim()].filter(Boolean).join(" · ") || "the library"}
          </button>
          {errors.map((e) => (
            <div key={e} className="error">
              {e}
            </div>
          ))}
          {inFlight.length > 0 && (
            <ul className="doc-list">
              {inFlight.map((u) => (
                <li key={u.key} className="busy" aria-busy="true">
                  <div className="doc-name ellipsis" title={u.name}>
                    <span className="spinner" aria-hidden="true" />
                    {u.name}
                  </div>
                  <div className="progress" role="progressbar" aria-valuenow={Math.round(u.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                    <div className="progress-fill" style={{ width: `${Math.max(3, u.progress * 100)}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="manager-card">
          <div className="manager-list-header">
            <h3>Books</h3>
            {books.length > 6 && (
              <input
                type="search"
                placeholder="Search"
                aria-label="Search the library"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            )}
          </div>
          {books.length === 0 && <p className="small muted">No books yet.</p>}
          {shelves(books.filter((d) => matches(d, query))).map((shelf) => (
            <details key={shelf.key} className="shelf" open>
              <summary>
                <span className="ellipsis">{shelf.label}</span>
                <span className="badge">{shelf.docs.length}</span>
              </summary>
              <ul className="doc-list">
                {shelf.docs.map((d) => (
                  <li key={d.id} className={isWorking(d) ? "busy" : undefined}>
                    <div className="doc-name ellipsis" title={d.filename}>
                      {isWorking(d) && <span className="spinner" aria-hidden="true" />}
                      {bookTitle(d)}
                    </div>
                    <div className="doc-meta">
                      <span className={`status ${d.status}`}>{d.status}</span>
                      <span className="small muted">
                        {d.status === "ready" ? `${d.chunk_count} chunks` : d.status === "failed" ? "" : "in the queue or being read…"}
                        {d.status === "ready" && d.figure_status === "processing" ? " · finding figures…" : ""}
                        {d.status === "ready" && d.figure_status === "done" ? ` · ${d.figure_count} figures` : ""}
                        {d.status === "ready" && (d.vision_pending || d.figure_status === "waiting")
                          ? ` · ${d.vision_pending ? "math pages & figures" : "figures"} waiting for AI limits (finishes automatically)`
                          : ""}
                      </span>
                      <span className="manager-actions">
                        {d.status === "ready" && !isWorking(d) && (
                          <button
                            className="link small"
                            title="Read the math pages and figures again (pages already read are reused)"
                            onClick={() => reprocess(d)}
                          >
                            Re-process
                          </button>
                        )}
                        <button className="link small" onClick={() => remove(d)}>
                          Delete
                        </button>
                      </span>
                    </div>
                    {d.error && <div className="error small">{d.error}</div>}
                  </li>
                ))}
              </ul>
            </details>
          ))}
        </section>
      </div>
    </div>
  );
}
