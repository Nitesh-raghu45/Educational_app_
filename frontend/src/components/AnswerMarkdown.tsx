import { ReactNode, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { Citation } from "../types";

// The model cites as [1], 【1】 or 【1†source】.
const CITATION_RE = /[[【](\d{1,2})(?:†[^\]】]*)?[\]】]/g;
const CITE_LINK = "#cite-";
// A run of citation links followed by sentence punctuation (incl. the Hindi danda "।").
const CITES_BEFORE_PUNCTUATION_RE = /\s*((?:\[\d{1,2}\]\(#cite-\d{1,2}\))+)\s*([.,;:!?।])/g;

/** Models often write LaTeX's \[ … \] and \( … \); the math plugin understands $$ … $$ and $ … $.
 * (Replacement functions return text literally, so "$" needs no escaping.) */
export function normalizeMath(markdown: string): string {
  return markdown
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, expr: string) => `\n$$\n${expr.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, expr: string) => `$${expr.trim()}$`);
}

/** Turn citation markers into links the renderer shows as chips, drop repeats of the same
 * number in a row, and move punctuation in front of them ("text [1] ." -> "text. [1]"). */
export function prepareCitations(markdown: string): string {
  let text = markdown.replace(CITATION_RE, (_, n: string) => `[${n}](${CITE_LINK}${n})`);
  text = text.replace(/(\[(\d{1,2})\]\(#cite-\2\))(?:\s*\[\2\]\(#cite-\2\))+/g, "$1");
  return text.replace(CITES_BEFORE_PUNCTUATION_RE, "$2 $1");
}

const POPOVER_WIDTH = 320;

function CitationChip({ n, citation }: { n: string; citation?: Citation }) {
  // Positioned against the window (position: fixed), so it isn't clipped by scrolling tables.
  const [pos, setPos] = useState<{ left: number; top: number; above: boolean } | null>(null);
  const show = (el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    const width = Math.min(POPOVER_WIDTH, window.innerWidth - 16);
    const left = Math.min(Math.max(r.left + r.width / 2 - width / 2, 8), window.innerWidth - width - 8);
    const above = r.top > 170;
    setPos({ left, top: above ? r.top - 8 : r.bottom + 8, above });
  };
  return (
    <span className="cite">
      <button
        type="button"
        className="cite-chip"
        aria-label={`Source ${n}${citation ? `: ${citation.filename}` : ""}`}
        onMouseEnter={(e) => show(e.currentTarget)}
        onFocus={(e) => show(e.currentTarget)}
        onMouseLeave={() => setPos(null)}
        onBlur={() => setPos(null)}
        onClick={(e) => (pos ? setPos(null) : show(e.currentTarget))}
      >
        {n}
      </button>
      {citation && pos && (
        <span
          className="cite-pop"
          role="tooltip"
          style={{ left: pos.left, top: pos.top, transform: pos.above ? "translateY(-100%)" : undefined }}
        >
          <span className="cite-pop-title">
            {citation.filename}
            {citation.page ? ` · page ${citation.page}` : ""}
          </span>
          <span className="cite-pop-snippet">{citation.snippet}</span>
        </span>
      )}
    </span>
  );
}

function CodeBlock({ children }: { children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="code-block">
      <button
        type="button"
        className="copy-code"
        onClick={(e) => {
          const code = (e.currentTarget.nextElementSibling as HTMLElement | null)?.innerText ?? "";
          navigator.clipboard?.writeText(code).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <pre>{children}</pre>
    </div>
  );
}

export default function AnswerMarkdown({ content, citations }: { content: string; citations: Citation[] | null }) {
  const byIndex = new Map((citations ?? []).map((c) => [String(c.index), c]));
  const components: Components = {
    a({ href, children }) {
      if (href?.startsWith(CITE_LINK)) {
        const n = href.slice(CITE_LINK.length);
        return <CitationChip n={n} citation={byIndex.get(n)} />;
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    },
    table({ children }) {
      return (
        <div className="table-wrap">
          <table>{children}</table>
        </div>
      );
    },
    pre({ children }) {
      return <CodeBlock>{children}</CodeBlock>;
    },
  };
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
      {prepareCitations(normalizeMath(content))}
    </ReactMarkdown>
  );
}
