import { useEffect, useState } from "react";
import { Catalog, NO_SCOPE, StudyScope, chapterName, describeScope, gradeLabel } from "../library";
import { BookIcon, ChevronIcon } from "./icons";

interface Props {
  catalog: Catalog;
  scope: StudyScope;
  onChange: (scope: StudyScope) => void;
  busy?: boolean;
  /** "bar": one line under the chat title (a summary button on phones); "card": the new-chat screen. */
  variant: "bar" | "card";
}

/** Class → Subject → Chapter, as native selects (the phone's own picker on mobile). */
export default function StudyPicker({ catalog, scope, onChange, busy, variant }: Props) {
  // A class picked before its subject: kept here until the subject makes the choice complete.
  const [draft, setDraft] = useState<StudyScope>(scope);
  const [open, setOpen] = useState(false);
  useEffect(() => setDraft(scope), [scope]);

  const grades = catalog.grades();
  const subjects = draft.grade ? catalog.subjects(draft.grade) : [];
  const chapters = draft.subject ? catalog.chapters(draft.grade, draft.subject) : [];
  const summary = describeScope(scope, catalog);

  function pickGrade(grade: string) {
    if (!grade) {
      setDraft(NO_SCOPE);
      if (scope.grade || scope.subject) onChange(NO_SCOPE);
      return;
    }
    const options = catalog.subjects(grade);
    // One subject for this class: nothing more to choose.
    if (options.length === 1) {
      onChange({ grade, subject: options[0], chapter: "all" });
      setOpen(false);
    } else setDraft({ grade, subject: "", chapter: "all" });
  }

  function pickSubject(subject: string) {
    if (!subject) return setDraft({ ...draft, subject: "", chapter: "all" });
    onChange({ grade: draft.grade, subject, chapter: "all" });
    setOpen(false);
  }

  function pickChapter(chapter: string) {
    onChange({ ...draft, chapter });
    setOpen(false);
  }

  const fields = (
    <div className={`picker-fields ${variant}`}>
      <label className="picker-field grade">
        <span>Class</span>
        <select value={draft.grade} onChange={(e) => pickGrade(e.target.value)} disabled={busy}>
          <option value="">{variant === "bar" && (scope.grade || scope.subject) ? "None" : "Choose…"}</option>
          {grades.map((g) => (
            <option key={g} value={g}>
              {gradeLabel(g) || "Other"}
            </option>
          ))}
        </select>
      </label>
      <label className="picker-field subject">
        <span>Subject</span>
        <select value={draft.subject} onChange={(e) => pickSubject(e.target.value)} disabled={busy || !draft.grade}>
          <option value="">Choose…</option>
          {subjects.map((s) => (
            <option key={s} value={s}>
              {s || "Other"}
            </option>
          ))}
        </select>
      </label>
      <label className="picker-field chapter">
        <span>Chapter</span>
        <select
          value={draft.subject ? draft.chapter : "all"}
          onChange={(e) => pickChapter(e.target.value)}
          disabled={busy || !draft.subject || chapters.length < 2}
        >
          <option value="all">{chapters.length ? `All chapters (${chapters.length})` : "All chapters"}</option>
          {chapters.map((c) => (
            <option key={c.id} value={c.id}>
              {chapterName(c)}
            </option>
          ))}
        </select>
      </label>
    </div>
  );

  if (variant === "card") return fields;

  return (
    <div className={`study-bar ${open ? "open" : ""}`}>
      {/* Phones: one tappable line that opens the three pickers. */}
      <button className="study-summary" onClick={() => setOpen((v) => !v)} aria-expanded={open} disabled={busy}>
        <BookIcon />
        <span className="ellipsis">{summary || "Choose class & subject"}</span>
        {busy ? <span className="spinner" aria-hidden="true" /> : <ChevronIcon />}
      </button>
      <div className="study-fields">
        <span className="study-label" aria-hidden="true">
          <BookIcon /> Studying
        </span>
        {fields}
        {busy && <span className="spinner" aria-label="Updating" />}
      </div>
    </div>
  );
}
