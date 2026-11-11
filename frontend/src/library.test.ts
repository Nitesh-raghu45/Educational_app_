import { describe, expect, it } from "vitest";
import { Catalog, NO_SCOPE, chapterName, describeScope, scopeOf } from "./library";
import type { DocumentInfo } from "./types";

function book(id: string, filename: string, grade: string, subject: string, extra: Partial<DocumentInfo> = {}): DocumentInfo {
  return {
    id, filename, grade, subject, library: true, status: "ready", content_type: "application/pdf", size_bytes: 1,
    error: null, chunk_count: 1, created_at: "", figure_status: "done", vision_pending: false, figure_count: 0, ...extra,
  };
}

const books = [
  book("m6", "Ch 6 TRIANGLES.pdf", "10", "Mathematics"),
  book("m10", "Ch 10 CIRCLES.pdf", "10", "Mathematics"),
  book("m2", "Ch 2 Polynomials.pdf", "10", "Mathematics"),
  book("s1", "Ch 1 Chemical Reactions and Equations.pdf", "10", "Science"),
  book("n1", "Ch 1 Number Systems.pdf", "9", "Mathematics"),
  book("busy", "Ch 3 Metals.pdf", "10", "Science", { status: "processing" }),
];
const catalog = new Catalog(books);

describe("chapter names", () => {
  it("reads like a chapter list", () => {
    expect(chapterName(books[0])).toBe("Ch 6 · Triangles");
    expect(chapterName(books[3])).toBe("Ch 1 · Chemical Reactions and Equations");
  });
});

describe("catalog", () => {
  it("lists classes, subjects and chapters in natural order, ready books only", () => {
    expect(catalog.grades()).toEqual(["9", "10"]);
    expect(catalog.subjects("10")).toEqual(["Mathematics", "Science"]);
    expect(catalog.chapters("10", "Mathematics").map((c) => c.id)).toEqual(["m2", "m6", "m10"]);
    expect(catalog.chapters("10", "Science").map((c) => c.id)).toEqual(["s1"]);
  });

  it("turns a scope into the chapters to study", () => {
    expect(catalog.documentIds({ grade: "10", subject: "Mathematics", chapter: "all" })).toEqual(["m2", "m6", "m10"]);
    expect(catalog.documentIds({ grade: "10", subject: "Mathematics", chapter: "m6" })).toEqual(["m6"]);
    expect(catalog.documentIds(NO_SCOPE)).toEqual([]);
  });
});

describe("a chat's scope", () => {
  it("comes from its library books, ignoring the student's own files", () => {
    const own = book("own", "my notes.pdf", "", "", { library: false });
    expect(scopeOf([own], catalog)).toEqual(NO_SCOPE);
    expect(scopeOf([own, books[0]], catalog)).toEqual({ grade: "10", subject: "Mathematics", chapter: "m6" });
    expect(scopeOf(books.slice(0, 3), catalog)).toEqual({ grade: "10", subject: "Mathematics", chapter: "all" });
  });

  it("is described for the picker and the message box", () => {
    expect(describeScope({ grade: "10", subject: "Mathematics", chapter: "m6" }, catalog)).toBe(
      "Class 10 · Mathematics · Ch 6 · Triangles",
    );
    expect(describeScope({ grade: "10", subject: "Science", chapter: "all" }, catalog)).toBe("Class 10 · Science");
  });
});
