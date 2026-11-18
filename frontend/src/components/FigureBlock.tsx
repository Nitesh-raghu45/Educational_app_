import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { Citation } from "../types";

// Object URLs by figure id, so re-renders and repeated answers don't download an image again.
const imageUrls = new Map<string, Promise<string>>();

function isGone(err: unknown): boolean {
  return err instanceof ApiError && err.status === 404;
}

function loadImage(id: string): Promise<string> {
  let url = imageUrls.get(id);
  if (!url) {
    url = api.figureImage(id).then((blob) => URL.createObjectURL(blob));
    // A network hiccup may pass, so a later render retries. A 404 won't: the image is gone (e.g. an
    // old answer whose figure was lost in a storage move), so it stays remembered as gone.
    url.catch((err) => !isGone(err) && imageUrls.delete(id));
    imageUrls.set(id, url);
  }
  return url;
}

/** A real figure from the student's document, shown where the answer refers to it. */
export default function FigureBlock({ citation }: { citation: Citation }) {
  const figure = citation.figure!;
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState<"gone" | "error" | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFailed(null);
    loadImage(figure.id).then(
      (url) => !cancelled && setSrc(url),
      (err) => !cancelled && setFailed(isGone(err) ? "gone" : "error"),
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

  if (failed === "gone") {
    // Re-uploading makes figures with new ids, so it can't bring this one back: say so briefly
    // instead of showing an empty frame.
    return <p className="figure-gone">Figure “{figure.title}” is no longer available.</p>;
  }

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
          <div className="visual-loading">This figure couldn't be loaded. Check your connection and reopen the chat.</div>
        ) : (
          <div className="visual-loading">
            <span className="spinner" aria-hidden="true" /> Loading figure…
          </div>
        )}
      </div>
      <figcaption>
        <span className="figure-title">{figure.title}</span>
      </figcaption>
    </figure>
  );
}
