export interface User {
  id: string;
  email: string;
}

export interface Section {
  id: string;
  title: string | null;
  question_count: number;
  max_questions: number;
  remaining: number;
  status: "active" | "limit_reached";
  created_at: string;
  updated_at: string;
}

export interface Citation {
  index: number | null;
  chunk_id: string;
  document_id: string;
  filename: string;
  page: number | null;
  snippet: string;
  score: number;
  /** Present when this source is a figure from the document, shown as an image. */
  figure?: { id: string; title: string; width: number; height: number };
  /** Present when this entry is a picture from the web shown as [[web:n]]: not a source. */
  web_image?: WebImage;
}

export interface WebImage {
  n: number;
  title: string;
  image_url: string;
  page_url: string;
  source: string;
  provider: "wikipedia" | "google";
  description: string;
  width: number | null;
  height: number | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  status: "pending" | "streaming" | "completed" | "failed";
  reply_to_id: string | null;
  created_at: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: "pending" | "processing" | "ready" | "failed";
  error: string | null;
  chunk_count: number;
  created_at: string;
  figure_status: "processing" | "done" | "failed" | null;
  /** Upload responses only: the same file was already in the library / already in this chat. */
  duplicate?: "existing" | "already_in_chat" | null;
  figure_count: number;
}

export interface Quiz {
  id: string;
  section_id: string;
  topic: string | null;
  created_at: string;
  questions: { question: string; options: string[] }[];
}

export interface QuestionResult {
  question: string;
  options: string[];
  selected: number | null;
  correct_index: number;
  is_correct: boolean;
  explanation: string;
  citation: Citation | null;
}

export interface Attempt {
  id: string;
  score: number;
  total: number;
  results: QuestionResult[];
}

export interface UsageItem {
  used: number;
  limit: number; // 0 = unlimited
}

export interface Usage {
  resets_at: string;
  limits: Record<"question" | "quiz" | "upload" | "chat", UsageItem>;
}

export interface StartEvent {
  section: Section;
  user_message: Message;
  assistant_message: Message;
}
