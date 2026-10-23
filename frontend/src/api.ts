import type { Attempt, DocumentInfo, Message, Quiz, Section, Usage, User } from "./types";

// Empty in dev/docker (same origin, proxied). On Render, VITE_API_URL is the API service URL.
const BASE = `${(import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "")}/api/v1`;
const TOKEN_KEY = "study.token";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

/** "You've reached today's limit…" plus when it resets, in the reader's local time. */
export function describeError(err: unknown, fallback: string): string {
  if (!(err instanceof ApiError)) return fallback;
  if (err.code === "DAILY_LIMIT_REACHED" && typeof err.details.resets_at === "string") {
    const resets = new Date(err.details.resets_at);
    const time = resets.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    return err.message.replace("It resets at midnight UTC.", `It resets at ${time} your time.`);
  }
  return err.message;
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: session-only login */
  }
}

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler;
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function toApiError(res: Response): Promise<ApiError> {
  if (res.status === 401) onUnauthorized();
  try {
    const body = await res.json();
    if (body?.error) {
      const { code, message, ...details } = body.error;
      return new ApiError(res.status, code, message, details);
    }
    if (Array.isArray(body?.detail)) {
      return new ApiError(res.status, "VALIDATION_ERROR", body.detail.map((d: { msg: string }) => d.msg).join("; "));
    }
  } catch {
    /* non-JSON body */
  }
  return new ApiError(res.status, "HTTP_ERROR", `Request failed (${res.status})`);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const isForm = body instanceof FormData;
  const res = await fetch(BASE + path, {
    method,
    headers: { ...authHeaders(), ...(body && !isForm ? { "Content-Type": "application/json" } : {}) },
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
  });
  if (!res.ok) throw await toApiError(res);
  return res.status === 204 ? (undefined as T) : res.json();
}

/**
 * POST and read a Server-Sent Events stream. Native EventSource can't send a POST body or
 * Authorization header, so this parses the stream from fetch() directly.
 */
export async function streamSSE(
  path: string,
  body: unknown,
  onEvent: (event: string, data: any) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json", Accept: "text/event-stream" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await toApiError(res);
  if (!res.body) throw new ApiError(0, "NO_STREAM", "Streaming is not supported by this browser.");

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith(":")) continue; // heartbeat comment
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  }
}

export const api = {
  register: (email: string, password: string) =>
    request<{ access_token: string; user: User }>("POST", "/auth/register", { email, password }),
  login: (email: string, password: string) =>
    request<{ access_token: string; user: User }>("POST", "/auth/login", { email, password }),
  me: () => request<User>("GET", "/auth/me"),
  usage: () => request<Usage>("GET", "/usage"),

  listSections: () => request<Section[]>("GET", "/sections"),
  createSection: () => request<Section>("POST", "/sections", {}),
  getSection: (id: string) => request<Section>("GET", `/sections/${id}`),
  renameSection: (id: string, title: string) => request<Section>("PATCH", `/sections/${id}`, { title }),
  deleteSection: (id: string) => request<void>("DELETE", `/sections/${id}`),
  listMessages: (id: string) => request<Message[]>("GET", `/sections/${id}/messages`),

  listDocuments: () => request<DocumentInfo[]>("GET", "/documents"),
  sectionDocuments: (id: string) => request<DocumentInfo[]>("GET", `/sections/${id}/documents`),
  /** Uses XMLHttpRequest because fetch() can't report upload progress. */
  uploadDocument: (file: File, sectionId: string, onProgress?: (fraction: number) => void) =>
    new Promise<DocumentInfo>((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      form.append("section_id", sectionId);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", BASE + "/documents");
      for (const [key, value] of Object.entries(authHeaders())) xhr.setRequestHeader(key, value);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total);
      };
      xhr.onload = async () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(await toApiError(new Response(xhr.responseText, { status: xhr.status })));
        }
      };
      xhr.onerror = () => reject(new ApiError(0, "NETWORK_ERROR", "Upload failed. Check your connection and try again."));
      xhr.send(form);
    }),
  attachDocument: (sectionId: string, documentId: string) =>
    request<DocumentInfo>("POST", `/sections/${sectionId}/documents`, { document_id: documentId }),
  detachDocument: (sectionId: string, documentId: string) =>
    request<void>("DELETE", `/sections/${sectionId}/documents/${documentId}`),

  createQuiz: (sectionId: string, numQuestions: number, topic: string) =>
    request<Quiz>("POST", `/sections/${sectionId}/quizzes`, { num_questions: numQuestions, topic: topic || null }),
  submitAttempt: (quizId: string, answers: (number | null)[]) =>
    request<Attempt>("POST", `/quizzes/${quizId}/attempts`, { answers }),
};
