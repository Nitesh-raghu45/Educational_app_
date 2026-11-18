import { useState } from "react";
import type { Message } from "../types";
import { hasVisual, stripCitations } from "../visuals/markdownPrep";
import AnswerMarkdown from "./AnswerMarkdown";
import { CheckIcon, CopyIcon, SparklesIcon } from "./icons";

interface Props {
  message: Message;
  onRetry?: () => void;
  /** Ask for a diagram/chart of this answer (offered on the latest answer only). */
  onVisualize?: () => void;
}

/** Ghost icon button in the message footer, as in the shadcn chatbot template. */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="msg-action"
      aria-label={copied ? "Copied" : "Copy"}
      title={copied ? "Copied" : "Copy"}
      onClick={() =>
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        })
      }
    >
      {copied ? <CheckIcon /> : <CopyIcon />}
    </button>
  );
}

export default function MessageBubble({ message, onRetry, onVisualize }: Props) {
  if (message.role === "user") {
    return (
      <div className="message user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const streaming = message.status === "pending" || message.status === "streaming";
  const waiting = streaming && !message.content;

  return (
    <div className="message assistant">
      <div className="answer">
        {waiting ? (
          <div className="thinking" role="status">
            <span className="spinner" aria-hidden="true" />
            <span className="thinking-label">Thinking</span>
          </div>
        ) : (
          <div className="markdown">
            <AnswerMarkdown content={message.content} citations={message.citations} streaming={streaming} />
            {message.status === "streaming" && <span className="cursor" />}
          </div>
        )}

        {message.status === "completed" && message.content && (
          <div className="msg-actions">
            <CopyButton text={stripCitations(message.content)} />
            {onVisualize && !hasVisual(message.content) && (
              <button
                type="button"
                className="msg-action"
                onClick={onVisualize}
                title="Draw a diagram or chart of this answer (uses 1 question)"
              >
                <SparklesIcon /> Visualize
              </button>
            )}
          </div>
        )}

        {message.status === "failed" && (
          <div className="failed">
            The answer couldn't be generated.
            {onRetry && (
              <button className="secondary small" onClick={onRetry}>
                Retry (won't use a question)
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
