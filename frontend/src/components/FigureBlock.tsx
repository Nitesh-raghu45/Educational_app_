import { useEffect, useState } from "react";
import { api } from "../api";
import type { Citation } from "../types";

// Object URLs by figure id, so re-renders and repeated answers don't download an image again.
const imageUrls = new Map<string, Promise<string>>();

function loadImage(id: string): Promise<string> {
  let url = imageUrls.get(id);
  if (!url) {
    url = api.figureImage(id).then((blob) => URL.createObjectURL(blob));
    url.catch(() => imageUrls.delete(id)); // let a later render retry
    imageUrls.set(id, url);
  }
  return url;
}

/** A real figure from the student's document, shown where the answer refers to it. */
export default function FigureBlock({ citation }: { citation: Citation }) {
  const figure = citation.figure!;
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFailed(false);
    loadImage(figure.id).then(
      (url) => !cancelled && setSrc(url),
      () => !cancelled && setFailed(true),
    );
    return () => {
      cancelled = true;
    };
  }, [figure.id]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const source = `${citation.filename}${citation.page ? ` · page ${citation.page}` : ""}`;
  return (
    <figure className={`visual doc-figure ${expanded ? "expanded" : ""}`}>
      {src && (
        <div className="visual-toolbar">
          <button type="button" onClick={() => setExpanded((v) => !v)} aria-label={expanded ? "Close full screen" : "Full screen"}>
            {expanded ? "Close" : "Expand"}
          </button>
        </div>
      )}
      <div className="figure-frame" style={{ aspectRatio: `${figure.width} / ${figure.height}` }}>
        {src ? (
          <img src={src} alt={figure.title} onClick={() => setExpanded(true)} />
        ) : failed ? (
          <div className="visual-loading">This figure couldn't be loaded. Upload the document again to restore it.</div>
        ) : (
          <div className="visual-loading">
            <span className="spinner" aria-hidden="true" /> Loading figure…
          </div>
        )}
      </div>
      <figcaption>
        <span className="figure-title">{figure.title}</span>
        <span className="figure-source">
          <span className="source-num">{citation.index}</span> {source}
        </span>
      </figcaption>
    </figure>
  );
}
