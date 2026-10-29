import { KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import { api, describeError, streamSSE } from "../api";
import type { DocumentInfo, Message, Section, StartEvent } from "../types";
import DocumentsPanel from "./DocumentsPanel";
import MessageBubble from "./MessageBubble";
import QuizPanel from "./QuizPanel";
import { DocumentIcon, QuizIcon, SendIcon } from "./icons";

// Phones and tablets: an on-screen keyboard, where Enter should add a line, not send.
const touchKeyboard = typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches;

interface Props {
  sectionId: string;
  onSectionChange: (section: Section) => void;
  onNewChat: () => void;
}

// Sent by the Visualize button. "previous answer" makes the backend resolve what to draw from
// the conversation, and the answer rules keep the visual grounded in the documents.
const VISUALIZE_REQUEST = "Explain your previous answer again in simple words, with diagrams or charts placed next to the parts they explain, using only my documents.";

function newClientId(): string {
  return crypto.randomUUID().replace(/-/g, "");
}

export default function ChatView({ sectionId, onSectionChange, onNewChat }: Props) {
  const [section, setSection] = useState<Section | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDocs, setShowDocs] = useState(false);
  const [showQuiz, setShowQuiz] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Follow a streaming answer only while the reader is at the bottom, so scrolling up to
  // reread something isn't interrupted by every new token.
  const followRef = useRef(true);

  const updateSection = useCallback(
    (s: Section) => {
      setSection(s);
      onSectionChange(s);
    },
    [onSectionChange],
  );

  const reloadMessages = useCallback(() => api.listMessages(sectionId).then(setMessages), [sectionId]);
  const reloadDocs = useCallback(() => api.sectionDocuments(sectionId).then(setDocs), [sectionId]);

  useEffect(() => {
    Promise.all([api.getSection(sectionId), api.listMessages(sectionId), api.sectionDocuments(sectionId)])
      .then(([s, m, d]) => {
        setSection(s);
        setMessages(m);
        setDocs(d);
        if (d.length === 0) setShowDocs(true);
      })
      .catch((err) => setError(describeError(err, "Couldn't load this chat.")));
    return () => abortRef.current?.abort(); // the server still finishes and saves the answer
  }, [sectionId]);

  // Poll while documents are being processed.
  useEffect(() => {
    const reading = docs.some((d) => d.status === "pending" || d.status === "processing");
    // Figures are found after the text is ready (questions already work), so check less often.
    const findingFigures = docs.some((d) => d.status === "ready" && d.figure_status === "processing");
    if (!reading && !findingFigures) return;
    const timer = setTimeout(reloadDocs, reading ? 1500 : 4000);
    return () => clearTimeout(timer);
  }, [docs, reloadDocs]);

  // An answer generating from a previous visit: poll until it settles.
  useEffect(() => {
    if (sending || !messages.some((m) => m.status === "pending" || m.status === "streaming")) return;
    const timer = setTimeout(reloadMessages, 2000);
    return () => clearTimeout(timer);
  }, [messages, sending, reloadMessages]);

  // Scroll the message list itself (scrollIntoView can also move the page on phones, e.g.
  // while the keyboard is open).
  useEffect(() => {
    const list = messagesRef.current;
    if (list && followRef.current) list.scrollTop = list.scrollHeight;
  }, [messages]);

  function onMessagesScroll() {
    const list = messagesRef.current;
    if (list) followRef.current = list.scrollHeight - list.scrollTop - list.clientHeight < 120;
  }

  // Grow the message box with its text (CSS field-sizing isn't supported by Safari/Firefox).
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
    // Scroll inside the box only once it is full height; before that it fits its text exactly.
    input.style.overflowY = input.scrollHeight > 200 ? "auto" : "hidden";
  }, [draft]);

  async function runStream(path: string, body: unknown, placeholder: { userId?: string; assistantId: string }) {
    let assistantId = placeholder.assistantId;
    let started = false;
    const patchAssistant = (fn: (m: Message) => Message) =>
      setMessages((ms) => ms.map((m) => (m.id === assistantId ? fn(m) : m)));

    setSending(true);
    setError(null);
    abortRef.current = new AbortController();
    try {
      await streamSSE(
        path,
        body,
        (event, data) => {
          if (event === "message_start") {
            const start = data as StartEvent;
            const placeholderId = assistantId;
            started = true;
            updateSection(start.section);
            setMessages((ms) =>
              ms.map((m) =>
                m.id === placeholder.userId
                  ? start.user_message
                  : m.id === placeholderId
                    ? { ...start.assistant_message, content: "", status: "streaming" }
                    : m,
              ),
            );
            assistantId = start.assistant_message.id;
          } else if (event === "token") {
            patchAssistant((m) => ({ ...m, status: "streaming", content: m.content + data.text }));
          } else if (event === "citations") {
            patchAssistant((m) => ({ ...m, citations: data.citations }));
          } else if (event === "done") {
            patchAssistant((m) => ({ ...m, status: "completed" }));
          } else if (event === "error") {
            patchAssistant((m) => ({ ...m, status: "failed" }));
          }
        },
        abortRef.current.signal,
      );
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      if (!started && placeholder.userId) {
        // Rejected before a slot was reserved (limit reached, no documents, rate limit…).
        setMessages((ms) => ms.filter((m) => m.id !== placeholder.userId && m.id !== placeholder.assistantId));
        setError(describeError(err, "Couldn't send your question."));
        api.getSection(sectionId).then(updateSection);
      } else if (!started) {
        setError(describeError(err, "Couldn't retry."));
        reloadMessages();
      } else {
        // Connection dropped mid-stream; the server keeps generating, so fetch the saved result.
        setError("Connection interrupted — reloading the answer.");
        setTimeout(reloadMessages, 1500);
      }
    } finally {
      setSending(false);
    }
  }

  async function send(text?: string) {
    const content = (text ?? draft).trim();
    if (!content || sending) return;
    const clientId = newClientId();
    const now = new Date().toISOString();
    const userId = `tmp-u-${clientId}`;
    const assistantId = `tmp-a-${clientId}`;
    if (text === undefined) setDraft("");
    followRef.current = true; // jump to your own question
    setMessages((ms) => [
      ...ms,
      { id: userId, role: "user", content, citations: null, status: "completed", reply_to_id: null, created_at: now },
      { id: assistantId, role: "assistant", content: "", citations: null, status: "pending", reply_to_id: userId, created_at: now },
    ]);
    await runStream(`/sections/${sectionId}/messages`, { content, client_message_id: clientId }, { userId, assistantId });
  }

  async function retry(messageId: string) {
    setMessages((ms) => ms.map((m) => (m.id === messageId ? { ...m, status: "pending", content: "", citations: null } : m)));
    await runStream(`/sections/${sectionId}/retry/${messageId}`, undefined, { assistantId: messageId });
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // On phones Enter adds a new line (like ChatGPT); the send button sends.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && !touchKeyboard) {
      e.preventDefault();
      send();
    }
  }

  if (!section) return <div className="center muted">{error ?? "Loading…"}</div>;

  const limitReached = section.status === "limit_reached";
  const hasReadyDoc = docs.some((d) => d.status === "ready");
  const processing = docs.some((d) => d.status === "pending" || d.status === "processing");

  return (
    <div className="chat">
      <header className="chat-header">
        <h2 className="ellipsis">{section.title ?? "New chat"}</h2>
        <div className="header-actions">
          <span className={`quota ${limitReached ? "full" : section.remaining <= 5 ? "low" : ""}`}>
            {section.question_count} / {section.max_questions} questions
          </span>
          <button
            className="secondary header-button"
            onClick={() => setShowDocs((v) => !v)}
            aria-label={`Documents (${docs.length})`}
            aria-expanded={showDocs}
          >
            <DocumentIcon />
            <span className="button-label">Documents</span>
            <span className="badge">{docs.length}</span>
          </button>
          <button
            className="secondary header-button"
            onClick={() => setShowQuiz(true)}
            disabled={!hasReadyDoc}
            aria-label="Quiz me"
          >
            <QuizIcon />
            <span className="button-label">Quiz me</span>
          </button>
        </div>
      </header>

      <div className="chat-body">
        <div className="messages" ref={messagesRef} onScroll={onMessagesScroll}>
          {messages.length === 0 && (
            <div className="empty-state">
              <h1>What do you want to learn?</h1>
              <p className="muted">
                Answers come only from the documents in this chat. You can ask up to {section.max_questions} questions here.
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble
              key={m.id}
              message={m}
              onRetry={!sending && m.role === "assistant" ? () => retry(m.id) : undefined}
              onVisualize={
                !sending && i === messages.length - 1 && m.role === "assistant" && section.status === "active"
                  ? () => send(VISUALIZE_REQUEST)
                  : undefined
              }
            />
          ))}
        </div>
        {showDocs && (
          <DocumentsPanel sectionId={sectionId} docs={docs} onDocsChange={setDocs} onClose={() => setShowDocs(false)} />
        )}
      </div>

      <footer className="composer-wrap">
        {error && <div className="error banner">{error}</div>}
        {limitReached ? (
          <div className="limit-banner">
            <strong>Question limit reached</strong> — this chat used all {section.max_questions} questions. Its history stays
            available.
            <button className="primary" onClick={onNewChat}>
              Start a new chat
            </button>
          </div>
        ) : !hasReadyDoc ? (
          <div className="limit-banner">
            {processing ? (
              <span className="busy-label">
                <span className="spinner" aria-hidden="true" />
                Processing your documents… you can ask questions as soon as one is ready.
              </span>
            ) : (
              "Upload a document to start asking questions."
            )}
            {!processing && (
              <button className="primary" onClick={() => setShowDocs(true)}>
                Upload documents
              </button>
            )}
          </div>
        ) : (
          <div className="composer">
            <textarea
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about your documents…"
              rows={1}
              maxLength={4000}
              disabled={sending}
              enterKeyHint={touchKeyboard ? "enter" : "send"}
            />
            <button className="send" onClick={() => send()} disabled={sending || !draft.trim()} aria-label="Send">
              <SendIcon />
            </button>
          </div>
        )}
        <div className="small muted hint">
          {limitReached ? "" : `${section.remaining} question${section.remaining === 1 ? "" : "s"} left in this chat`}
        </div>
      </footer>

      {showQuiz && <QuizPanel sectionId={sectionId} onClose={() => setShowQuiz(false)} />}
    </div>
  );
}
