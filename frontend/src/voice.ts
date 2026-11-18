/** Speech to text with the browser's own Web Speech API (free, no package). Chrome, Edge and
 * Safari support it; elsewhere `voiceSupported` is false and the mic button is hidden. Chrome and
 * Edge send the audio to their speech service to transcribe it. */
import { useCallback, useEffect, useRef, useState } from "react";

// The API isn't in TypeScript's DOM types yet: only the parts used here.
interface Recognition {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
}
type RecognitionCtor = new () => Recognition;

const Ctor: RecognitionCtor | undefined =
  typeof window === "undefined"
    ? undefined
    : ((window as unknown as { SpeechRecognition?: RecognitionCtor; webkitSpeechRecognition?: RecognitionCtor })
        .SpeechRecognition ??
      (window as unknown as { webkitSpeechRecognition?: RecognitionCtor }).webkitSpeechRecognition);

export const voiceSupported = Boolean(Ctor);

export type VoiceLang = "en-IN" | "hi-IN";
export const VOICE_LANGS: { value: VoiceLang; label: string }[] = [
  { value: "en-IN", label: "English" },
  { value: "hi-IN", label: "हिंदी (Hindi)" },
];

const LANG_KEY = "voiceLang";

function savedLang(): VoiceLang {
  try {
    const value = localStorage.getItem(LANG_KEY);
    return value === "hi-IN" ? "hi-IN" : "en-IN";
  } catch {
    return "en-IN";
  }
}

const ERRORS: Record<string, string> = {
  "not-allowed": "Microphone access is blocked. Allow it in your browser's site settings.",
  "service-not-allowed": "Microphone access is blocked. Allow it in your browser's site settings.",
  "audio-capture": "No microphone was found.",
  network: "Voice typing needs an internet connection.",
};

/**
 * Dictation into the message box. `start(base)` keeps `base` (what was already typed) and adds
 * what is said after it, updating live through `onText`.
 */
export function useVoice(onText: (text: string) => void) {
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lang, setLangState] = useState<VoiceLang>(savedLang);
  const recognition = useRef<Recognition | null>(null);
  const onTextRef = useRef(onText);
  onTextRef.current = onText;

  const stop = useCallback(() => recognition.current?.stop(), []);

  const start = useCallback(
    (base: string) => {
      if (!Ctor || recognition.current) return;
      const rec = new Ctor();
      rec.lang = lang;
      rec.continuous = true; // keep listening through pauses until stopped
      rec.interimResults = true; // show words as they are recognised
      const prefix = base.trim() ? `${base.trimEnd()} ` : "";
      rec.onresult = (e) => {
        let said = "";
        for (let i = 0; i < e.results.length; i++) said += e.results[i][0].transcript;
        onTextRef.current(prefix + said.trimStart());
      };
      rec.onerror = (e) => {
        if (e.error !== "no-speech" && e.error !== "aborted") setError(ERRORS[e.error] ?? "Voice typing stopped.");
      };
      rec.onend = () => {
        recognition.current = null;
        setListening(false);
      };
      setError(null);
      recognition.current = rec;
      try {
        rec.start();
        setListening(true);
      } catch {
        recognition.current = null;
      }
    },
    [lang],
  );

  const setLang = useCallback((value: VoiceLang) => {
    setLangState(value);
    try {
      localStorage.setItem(LANG_KEY, value);
    } catch {
      /* private mode: the choice lasts for this visit */
    }
  }, []);

  useEffect(() => () => recognition.current?.abort(), []);

  return { listening, error, lang, setLang, start, stop, clearError: () => setError(null) };
}
