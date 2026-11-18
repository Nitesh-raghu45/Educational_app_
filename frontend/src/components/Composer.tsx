import { ChangeEvent, KeyboardEvent, RefObject, useEffect, useRef, useState } from "react";
import { VOICE_LANGS, useVoice, voiceSupported } from "../voice";
import { ACCEPT } from "./DocumentsPanel";
import {
  CameraIcon,
  CheckIcon,
  DocumentIcon,
  LanguagesIcon,
  MicIcon,
  PaperclipIcon,
  PlusIcon,
  QuizIcon,
  SendIcon,
  StopIcon,
} from "./icons";

const touchScreen = typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches;
const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

interface Props {
  inputRef: RefObject<HTMLTextAreaElement>;
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  placeholder: string;
  sending: boolean;
  enterKeyHint: "enter" | "send";
  onUpload: (e: ChangeEvent<HTMLInputElement>) => void;
  filesCount: number;
  onOpenFiles: () => void;
  onQuiz: () => void;
  quizDisabled: boolean;
  remaining: number;
  maxQuestions: number;
}

/** The message box: + menu (files, quiz, voice language), the text, a mic and a send button. */
export default function Composer(props: Props) {
  const { inputRef, value, onChange, sending } = props;
  const [menuOpen, setMenuOpen] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const voice = useVoice(onChange);

  // Close the menu on a click outside it or Escape.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => !wrapRef.current?.contains(e.target as Node) && setMenuOpen(false);
    const onKey = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setMenuOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  // Ctrl+U / ⌘U: add files, as the menu says.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "u") {
        e.preventDefault();
        fileRef.current?.click();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  // Sending stops dictation, so the next words don't land in the sent message.
  useEffect(() => {
    if (sending && voice.listening) voice.stop();
  }, [sending, voice]);

  function pick(action: () => void) {
    setMenuOpen(false);
    action();
  }

  function send() {
    if (voice.listening) voice.stop();
    props.onSend();
  }

  const hasText = Boolean(value.trim());

  return (
    <div className="composer-wrap-inner" ref={wrapRef}>
      <input ref={fileRef} type="file" accept={ACCEPT} multiple hidden onChange={props.onUpload} />
      <input ref={cameraRef} type="file" accept="image/*" capture="environment" hidden onChange={props.onUpload} />

      {menuOpen && (
        <div className="composer-menu" role="menu">
          <button role="menuitem" onClick={() => pick(() => fileRef.current?.click())}>
            <PaperclipIcon /> <span>Add files or photos</span>
            {!touchScreen && <kbd>{isMac ? "⌘U" : "Ctrl+U"}</kbd>}
          </button>
          {touchScreen && (
            <button role="menuitem" onClick={() => pick(() => cameraRef.current?.click())}>
              <CameraIcon /> <span>Take a photo</span>
            </button>
          )}
          <button role="menuitem" onClick={() => pick(props.onOpenFiles)}>
            <DocumentIcon /> <span>My files</span>
            {props.filesCount > 0 && <span className="badge">{props.filesCount}</span>}
          </button>
          <button role="menuitem" onClick={() => pick(props.onQuiz)} disabled={props.quizDisabled}>
            <QuizIcon /> <span>Quiz me</span>
          </button>
          {voiceSupported && (
            <>
              <div className="composer-menu-sep" />
              <div className="composer-menu-label">
                <LanguagesIcon /> Voice language
              </div>
              {VOICE_LANGS.map((l) => (
                <button
                  key={l.value}
                  role="menuitemradio"
                  aria-checked={voice.lang === l.value}
                  className="indent"
                  onClick={() => pick(() => voice.setLang(l.value))}
                >
                  <span>{l.label}</span>
                  {voice.lang === l.value && <CheckIcon />}
                </button>
              ))}
            </>
          )}
          <div className="composer-menu-sep" />
          <div className={`composer-menu-foot ${props.remaining <= 5 ? "low" : ""}`}>
            {props.remaining} of {props.maxQuestions} questions left in this chat
          </div>
        </div>
      )}

      <div className={`composer ${voice.listening ? "listening" : ""}`}>
        <button
          type="button"
          className="composer-plus"
          aria-label="Add files, quiz and more"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((v) => !v)}
        >
          <PlusIcon />
        </button>
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={props.onKeyDown}
          placeholder={voice.listening ? "Listening… speak now" : props.placeholder}
          rows={1}
          maxLength={4000}
          disabled={sending}
          enterKeyHint={props.enterKeyHint}
        />
        {voiceSupported && (
          <button
            type="button"
            className={`composer-mic ${voice.listening ? "on" : ""}`}
            aria-label={voice.listening ? "Stop voice typing" : "Voice typing"}
            title={voice.listening ? "Stop" : `Voice typing (${VOICE_LANGS.find((l) => l.value === voice.lang)?.label})`}
            onClick={() => (voice.listening ? voice.stop() : voice.start(value))}
            disabled={sending}
          >
            {voice.listening ? <StopIcon /> : <MicIcon />}
          </button>
        )}
        {(hasText || !voiceSupported) && (
          <button className="send" onClick={send} disabled={sending || !hasText} aria-label="Send message">
            <SendIcon />
          </button>
        )}
      </div>

      {voice.error && (
        <div className="composer-note error" role="alert">
          {voice.error}
          <button className="link" onClick={voice.clearError}>
            Dismiss
          </button>
        </div>
      )}
      {!voice.error && props.remaining <= 5 && (
        <div className="composer-note low">
          {props.remaining} question{props.remaining === 1 ? "" : "s"} left in this chat
        </div>
      )}
    </div>
  );
}
