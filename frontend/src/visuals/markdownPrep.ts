/** Text clean-up applied to answers before Markdown rendering, never inside code. */

// Fenced code blocks (``` or ~~~, closed or, while streaming, still open) and inline `code`.
const FENCE_RE = /(^|\n)(```|~~~)[^\n]*(?:\n[\s\S]*?)?(?:\n\2[ \t]*(?=\n|$)|$(?![\s\S]))/g;
const INLINE_CODE_RE = /(`[^`\n]*`)/g;

/** Apply `transform` to everything except code, so chart JSON like "values": [12, 30] or
 * Mermaid labels like A["x"] are never rewritten. */
export function mapOutsideCode(markdown: string, transform: (text: string) => string): string {
  let out = "";
  let last = 0;
  for (const match of markdown.matchAll(FENCE_RE)) {
    const start = (match.index ?? 0) + match[1].length;
    out += transformProse(markdown.slice(last, start), transform) + markdown.slice(start, (match.index ?? 0) + match[0].length);
    last = (match.index ?? 0) + match[0].length;
  }
  return out + transformProse(markdown.slice(last), transform);
}

function transformProse(text: string, transform: (text: string) => string): string {
  return text
    .split(INLINE_CODE_RE)
    .map((part, i) => (i % 2 === 1 ? part : transform(part)))
    .join("");
}

// The model cites as [1], 【1】 or 【1†source】.
export const CITATION_RE = /[[【](\d{1,2})(?:†[^\]】]*)?[\]】]/g;
export const CITE_LINK = "#cite-";
// A run of citation links followed by sentence punctuation (incl. the Hindi danda "।").
const CITES_BEFORE_PUNCTUATION_RE = /\s*((?:\[\d{1,2}\]\(#cite-\d{1,2}\))+)\s*([.,;:!?।])/g;

/** Models often write LaTeX's \[ … \] and \( … \); the math plugin understands $$ … $$ and $ … $.
 * (Replacement functions return text literally, so "$" needs no escaping.) */
export function normalizeMath(text: string): string {
  return text
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, expr: string) => `\n$$\n${expr.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, expr: string) => `$${expr.trim()}$`);
}

/** Turn citation markers into links the renderer shows as chips, drop repeats of the same
 * number in a row, and move punctuation in front of them ("text [1] ." -> "text. [1]"). */
export function prepareCitations(text: string): string {
  let out = text.replace(CITATION_RE, (_, n: string) => `[${n}](${CITE_LINK}${n})`);
  out = out.replace(/(\[(\d{1,2})\]\(#cite-\2\))(?:\s*\[\2\]\(#cite-\2\))+/g, "$1");
  return out.replace(CITES_BEFORE_PUNCTUATION_RE, "$2 $1");
}

// The model shows a figure from the documents as [[figure:N]], N being its source number.
export const FIGURE_REF_RE = /\[\[\s*figure\s*:\s*(\d{1,2})\s*\]\]/gi;

/** Turn figure markers into ```figure blocks (on their own lines), rendered as the image. */
export function prepareFigures(text: string): string {
  return text.replace(FIGURE_REF_RE, (_, n: string) => "\n\n```figure\n" + n + "\n```\n\n");
}

export function prepareAnswer(markdown: string): string {
  return mapOutsideCode(markdown, (text) => prepareCitations(normalizeMath(prepareFigures(text))));
}

/** Plain text for the Copy button: citation and figure markers removed (outside code only). */
export function stripCitations(markdown: string): string {
  return mapOutsideCode(markdown, (text) => text.replace(FIGURE_REF_RE, "").replace(CITATION_RE, ""))
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Language of a fenced block the renderer handles as a visual, if any. */
export const VISUAL_LANGUAGES = ["mermaid", "chart", "figure"] as const;
export type VisualLanguage = (typeof VISUAL_LANGUAGES)[number];

export function hasVisual(markdown: string): boolean {
  return /(^|\n)(```|~~~)\s*(mermaid|chart)\b/.test(markdown) || new RegExp(FIGURE_REF_RE.source, "i").test(markdown);
}
