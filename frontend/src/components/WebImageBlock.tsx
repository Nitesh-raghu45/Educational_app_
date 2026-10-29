import { useEffect, useState } from "react";
import type { WebImage } from "../types";

/** A picture found on the web for this topic: an illustration, labelled with where it comes from. */
export default function WebImageBlock({ image }: { image: WebImage }) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  // A web picture can disappear or block embedding; then the answer simply goes without it.
  if (failed) return null;

  const ratio = image.width && image.height ? `${image.width} / ${image.height}` : undefined;
  return (
    <figure className={`visual doc-figure web-image ${expanded ? "expanded" : ""}`}>
      {loaded && (
        <div className="visual-toolbar">
          <button type="button" onClick={() => setExpanded((v) => !v)} aria-label={expanded ? "Close full screen" : "Full screen"}>
            {expanded ? "Close" : "Expand"}
          </button>
        </div>
      )}
      <div className="figure-frame" style={{ aspectRatio: ratio }}>
        {/* no-referrer: the image's site isn't told which page the student was on. */}
        <img
          src={image.image_url}
          alt={image.title}
          referrerPolicy="no-referrer"
          loading="lazy"
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          onClick={() => setExpanded(true)}
        />
      </div>
      <figcaption>
        <span className="figure-title">{image.title}</span>
        <a className="figure-source" href={image.page_url} target="_blank" rel="noopener noreferrer">
          From the web · {image.source}
        </a>
      </figcaption>
    </figure>
  );
}
