import { isValidElement, lazy, ReactNode, Suspense, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { Citation } from "../types";
import { CITE_LINK, prepareAnswer } from "../visuals/markdownPrep";
import ChartBlock from "./ChartBlock";

// The diagram renderer pulls in Mermaid, so it is only downloaded when a diagram appears.
const MermaidBlock = lazy(() => import("./MermaidBlock"));

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

/** Language and text of a fenced code block, from react-markdown's <pre><code class="language-x"> */
function codeInfo(children: ReactNode): { language: string | null; text: string } {
  const child = Array.isArray(children) ? children[0] : children;
  if (!isValidElement<{ className?: string; children?: ReactNode }>(child)) return { language: null, text: "" };
  const language = /language-([\w-]+)/.exec(child.props.className ?? "")?.[1] ?? null;
  const text = String(child.props.children ?? "").replace(/\n$/, "");
  return { language, text };
}

function VisualPlaceholder({ kind }: { kind: string }) {
  return (
    <div className="visual visual-loading">
      <span className="spinner" aria-hidden="true" /> {kind === "chart" ? "Preparing chart…" : "Drawing diagram…"}
    </div>
  );
}

interface Props {
  content: string;
  citations: Citation[] | null;
  /** While the answer streams, visual blocks are incomplete: show a placeholder instead. */
  streaming?: boolean;
}

export default function AnswerMarkdown({ content, citations, streaming = false }: Props) {
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
      const { language, text } = codeInfo(children);
      if (language === "mermaid" || language === "chart") {
        if (streaming) return <VisualPlaceholder kind={language} />;
        return language === "chart" ? (
          <ChartBlock source={text} />
        ) : (
          <Suspense fallback={<VisualPlaceholder kind="mermaid" />}>
            <MermaidBlock source={text} />
          </Suspense>
        );
      }
      return <CodeBlock>{children}</CodeBlock>;
    },
  };
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
      {prepareAnswer(content)}
    </ReactMarkdown>
  );
}
