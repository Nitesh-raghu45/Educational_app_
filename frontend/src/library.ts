import { api } from "./api";
import type { DocumentInfo } from "./types";

export interface Shelf {
  key: string;
  label: string;
  docs: DocumentInfo[];
}

const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

/** "Ch 2 Polynomials.pdf" → "Ch 2 Polynomials" */
export function bookTitle(doc: DocumentInfo): string {
  return doc.filename.replace(/\.[a-z0-9]+$/i, "");
}

/** A chapter's name for students: "Ch 6 TRIANGLES.pdf" → "Ch 6 · Triangles". */
export function chapterName(doc: DocumentInfo): string {
  const title = bookTitle(doc)
    .replace(/[_]+/g, " ")
    .replace(/\b([A-Z]{4,})\b/g, (word) => word[0] + word.slice(1).toLowerCase())
    .replace(/\s+/g, " ")
    .trim();
  const numbered = title.match(/^(?:ch(?:apter)?\.?\s*)(\d+)\s*[-:.]?\s*(.*)$/i);
  return numbered ? `Ch ${numbered[1]}${numbered[2] ? ` · ${numbered[2]}` : ""}` : title;
}

export function gradeLabel(grade: string): string {
  return /^\d+$/.test(grade) ? `Class ${grade}` : grade;
}

/** Library books grouped by class and subject ("Class 10 · Mathematics"), classes in number order. */
export function shelves(docs: DocumentInfo[]): Shelf[] {
  const groups = new Map<string, Shelf>();
  for (const doc of docs) {
    const grade = doc.grade?.trim() ?? "";
    const subject = doc.subject?.trim() ?? "";
    const key = `${grade}\u0000${subject}`;
    const label = [grade && gradeLabel(grade), subject].filter(Boolean).join(" · ") || "Other";
    if (!groups.has(key)) groups.set(key, { key, label, docs: [] });
    groups.get(key)!.docs.push(doc);
  }
  for (const shelf of groups.values()) shelf.docs.sort((a, b) => natural.compare(a.filename, b.filename));
  return [...groups.values()].sort((a, b) => natural.compare(a.key, b.key));
}

export function matches(doc: DocumentInfo, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return [doc.filename, doc.subject ?? "", doc.grade ? `class ${doc.grade}` : ""].some((s) => s.toLowerCase().includes(q));
}

/* ---------- What a chat studies: class, subject and chapter ---------- */

export interface StudyScope {
  grade: string;
  subject: string;
  /** A chapter's document id, or "all". */
  chapter: string;
}

export const NO_SCOPE: StudyScope = { grade: "", subject: "", chapter: "all" };

/** The library as class → subject → chapters, for the pickers (ready books only). */
export class Catalog {
  constructor(readonly books: DocumentInfo[]) {
    this.books = books.filter((b) => b.status === "ready");
  }

  grades(): string[] {
    return unique(this.books.map((b) => b.grade?.trim() ?? "")).sort(natural.compare);
  }

  subjects(grade: string): string[] {
    return unique(this.books.filter((b) => (b.grade?.trim() ?? "") === grade).map((b) => b.subject?.trim() ?? "")).sort(
      natural.compare,
    );
  }

  chapters(grade: string, subject: string): DocumentInfo[] {
    return this.books
      .filter((b) => (b.grade?.trim() ?? "") === grade && (b.subject?.trim() ?? "") === subject)
      .sort((a, b) => natural.compare(a.filename, b.filename));
  }

  /** The library chapters a scope stands for. */
  documentIds(scope: StudyScope): string[] {
    if (!scope.subject && !scope.grade) return [];
    const chapters = this.chapters(scope.grade, scope.subject);
    if (scope.chapter !== "all") return chapters.some((c) => c.id === scope.chapter) ? [scope.chapter] : [];
    return chapters.map((c) => c.id);
  }
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

/** The scope a chat's attached library books add up to. */
export function scopeOf(docs: DocumentInfo[], catalog: Catalog): StudyScope {
  const books = docs.filter((d) => d.library);
  if (books.length === 0) return NO_SCOPE;
  const grade = books[0].grade?.trim() ?? "";
  const subject = books[0].subject?.trim() ?? "";
  const chapters = catalog.chapters(grade, subject);
  const chapter = books.length === 1 && chapters.length > 1 ? books[0].id : "all";
  return { grade, subject, chapter };
}

/** "Class 10 · Mathematics · Ch 6 · Triangles", for the picker's summary and the message box. */
export function describeScope(scope: StudyScope, catalog: Catalog): string {
  if (!scope.subject && !scope.grade) return "";
  const chapter = scope.chapter === "all" ? null : catalog.books.find((b) => b.id === scope.chapter);
  return [scope.grade && gradeLabel(scope.grade), scope.subject, chapter ? chapterName(chapter) : ""]
    .filter(Boolean)
    .join(" · ");
}

// The library changes rarely: fetched once per page load (and again after a minute).
let cached: { at: number; books: Promise<DocumentInfo[]> } | null = null;

export function loadLibrary(): Promise<DocumentInfo[]> {
  if (!cached || Date.now() - cached.at > 60_000) {
    const books = api.listLibrary().catch((err) => {
      cached = null;
      throw err;
    });
    cached = { at: Date.now(), books };
  }
  return cached.books;
}
