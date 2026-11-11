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

/** Library books grouped by class and subject ("Class 10 · Mathematics"), classes in number order. */
export function shelves(docs: DocumentInfo[]): Shelf[] {
  const groups = new Map<string, Shelf>();
  for (const doc of docs) {
    const grade = doc.grade?.trim() ?? "";
    const subject = doc.subject?.trim() ?? "";
    const key = `${grade}\u0000${subject}`;
    const label = [grade && (/^\d+$/.test(grade) ? `Class ${grade}` : grade), subject].filter(Boolean).join(" · ") || "Other";
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
