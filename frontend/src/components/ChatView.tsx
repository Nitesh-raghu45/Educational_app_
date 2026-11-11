import { Dispatch, KeyboardEvent, SetStateAction, useCallback, useEffect, useRef, useState } from "react";
import { api, describeError, streamSSE } from "../api";
import { Catalog, NO_SCOPE, StudyScope, describeScope, loadLibrary, scopeOf } from "../library";
import type { DocumentInfo, Message, Section, StartEvent } from "../types";
import DocumentsPanel, { UploadButtons, UploadProgress, useUploads } from "./DocumentsPanel";
import MessageBubble from "./MessageBubble";
import QuizPanel from "./QuizPanel";
import StudyPicker from "./StudyPicker";
import { BookIcon, PaperclipIcon, QuizIcon, SendIcon } from "./icons";

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
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [scopeBusy, setScopeBusy] = useState(false);
  const syncedRef = useRef(false);
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
      })
      .catch((err) => setError(describeError(err, "Couldn't load this chat.")));
    return () => abortRef.current?.abort(); // the server still finishes and saves the answer
  }, [sectionId]);

  useEffect(() => {
    loadLibrary()
      .then((books) => setCatalog(new Catalog(books)))
      .catch(() => setCatalog(new Catalog([])));
  }, []);

  const scope = catalog ? scopeOf(docs, catalog) : NO_SCOPE;

  async function applyScope(next: StudyScope) {
    if (!catalog) return;
    setScopeBusy(true);
    setError(null);
    try {
      setDocs(await api.setLibraryDocuments(sectionId, catalog.documentIds(next)));
    } catch (err) {
      setError(describeError(err, "Couldn't change the subject."));
    } finally {
      setScopeBusy(false);
    }
  }

  // "All chapters" of a subject: chapters added to the library since then join the chat too.
  useEffect(() => {
    if (!catalog || syncedRef.current || scope.chapter !== "all" || !scope.subject) return;
    syncedRef.current = true;
    const attached = new Set(docs.filter((d) => d.library).map((d) => d.id));
    if (catalog.documentIds(scope).some((id) => !attached.has(id))) applyScope(scope);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalog, docs]);

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
  const ownDocs = docs.filter((d) => !d.library);
  const studying = describeScope(scope, catalog ?? new Catalog([]));
  const hasLibrary = (catalog?.books.length ?? 0) > 0;
  const fresh = messages.length === 0 && docs.length === 0;
  const topic = scope.chapter !== "all" ? studying.split(" · ").slice(2).join(" · ") : scope.subject;
  const placeholder = topic ? `Ask about ${topic}…` : ownDocs.length ? "Ask about your files…" : "Ask a question…";

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
            aria-label={`My files (${ownDocs.length})`}
            aria-expanded={showDocs}
          >
            <PaperclipIcon />
            <span className="button-label">My files</span>
            {ownDocs.length > 0 && <span className="badge">{ownDocs.length}</span>}
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
      {hasLibrary && !fresh && catalog && (
        <StudyPicker catalog={catalog} scope={scope} onChange={applyScope} busy={scopeBusy} variant="bar" />
      )}

      <div className="chat-body">
        <div className="messages" ref={messagesRef} onScroll={onMessagesScroll}>
          {fresh ? (
            <Welcome
              sectionId={sectionId}
              catalog={hasLibrary ? catalog : null}
              scope={scope}
              busy={scopeBusy}
              onScope={applyScope}
              onDocsChange={setDocs}
            />
          ) : (
            messages.length === 0 && (
              <div className="empty-state">
                <h1>{topic ? `Let's study ${topic}` : "What do you want to know?"}</h1>
                <p className="muted">Ask anything, or start with one of these:</p>
                <div className="suggestions">
                  {suggestionsFor(scope, ownDocs.length > 0).map((q) => (
                    <button key={q} className="suggestion" onClick={() => send(q)} disabled={sending || !hasReadyDoc}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )
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
            ) : hasLibrary ? (
              "Choose your class and subject, or upload your own file, to start."
            ) : (
              "Upload a file to start asking questions."
            )}
            {!processing && !fresh && (
              <button className="primary" onClick={() => setShowDocs(true)}>
                Upload a file
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
              placeholder={placeholder}
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

function suggestionsFor(scope: StudyScope, ownFiles: boolean): string[] {
  if (scope.chapter !== "all")
    return ["Explain this chapter in simple words", "What are the important formulas and definitions?", "Solve an example step by step"];
  if (scope.subject) return ["What are the main topics in these chapters?", "Give me the important definitions", "Explain the first chapter simply"];
  if (ownFiles) return ["Summarise my file", "Explain it in simple words", "What are the key points?"];
  return [];
}

interface WelcomeProps {
  sectionId: string;
  catalog: Catalog | null;
  scope: StudyScope;
  busy: boolean;
  onScope: (scope: StudyScope) => void;
  onDocsChange: Dispatch<SetStateAction<DocumentInfo[]>>;
}

/** A new chat: study a subject from the library, or ask about your own file. */
function Welcome({ sectionId, catalog, scope, busy, onScope, onDocsChange }: WelcomeProps) {
  const { inFlight, error, upload } = useUploads(sectionId, onDocsChange);
  return (
    <div className="welcome">
      <h1>What do you want to study?</h1>
      <div className="welcome-cards">
        {catalog && (
          <section className="welcome-card">
            <div className="welcome-icon">
              <BookIcon />
            </div>
            <h2>Study a subject</h2>
            <p className="muted small">Pick your class and subject. Choose one chapter, or ask across all of them.</p>
            <StudyPicker catalog={catalog} scope={scope} onChange={onScope} busy={busy} variant="card" />
          </section>
        )}
        <section className="welcome-card">
          <div className="welcome-icon">
            <PaperclipIcon />
          </div>
          <h2>Ask about your own notes</h2>
          <p className="muted small">Upload a PDF, Word file or a photo of your notes or homework.</p>
          <UploadButtons onUpload={upload} />
          {error && <div className="error small">{error}</div>}
          {inFlight.length > 0 && (
            <ul className="doc-list">
              <UploadProgress inFlight={inFlight} />
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
