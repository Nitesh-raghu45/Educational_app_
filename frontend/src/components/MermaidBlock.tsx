import { useEffect, useId, useState } from "react";

// Mermaid is large (~1 MB), so it is downloaded only when an answer contains a diagram.
let mermaidPromise: Promise<typeof import("mermaid").default> | null = null;

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then(({ default: mermaid }) => {
      const dark = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict", // no HTML labels, scripts or click handlers from diagram text
        theme: dark ? "dark" : "default",
        fontFamily: 'system-ui, "Segoe UI", Roboto, "Nirmala UI", "Noto Sans Devanagari", sans-serif',
        suppressErrorRendering: true,
        flowchart: { useMaxWidth: true, htmlLabels: false, curve: "basis" },
        sequence: { useMaxWidth: true },
      });
      return mermaid;
    });
  }
  return mermaidPromise;
}

export default function MermaidBlock({ source }: { source: string }) {
  const id = `mermaid-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [showCode, setShowCode] = useState(false);

  // Esc closes the full-screen view.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  useEffect(() => {
    let cancelled = false;
    setSvg(null);
    setError(null);
    loadMermaid()
      .then(async (mermaid) => {
        await mermaid.parse(source); // throws a readable error for invalid diagram code
        const { svg } = await mermaid.render(id, source);
        if (!cancelled) setSvg(svg);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message.split("\n")[0] : "Invalid diagram");
      });
    return () => {
      cancelled = true;
    };
  }, [source, id]);

  if (error) {
    return (
      <div className="visual visual-error">
        <div className="visual-error-note">Couldn't draw this diagram, so here is its outline instead.</div>
        <pre>
          <code>{source}</code>
        </pre>
      </div>
    );
  }

  return (
    <figure className={`visual diagram ${expanded ? "expanded" : ""}`}>
      <div className="visual-toolbar">
        <button type="button" onClick={() => setShowCode((v) => !v)}>
          {showCode ? "Diagram" : "Code"}
        </button>
        <button type="button" onClick={() => setExpanded((v) => !v)} aria-label={expanded ? "Close full screen" : "Full screen"}>
          {expanded ? "Close" : "Expand"}
        </button>
      </div>
      {showCode ? (
        <pre className="diagram-code">
          <code>{source}</code>
        </pre>
      ) : svg ? (
        // Rendered by Mermaid with securityLevel "strict", which sanitises labels.
        <div className="diagram-svg" dangerouslySetInnerHTML={{ __html: svg }} />
      ) : (
        <div className="visual-loading">
          <span className="spinner" aria-hidden="true" /> Drawing diagram…
        </div>
      )}
    </figure>
  );
}
