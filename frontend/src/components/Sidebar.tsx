import type { Section, Usage, User } from "../types";

const USAGE_ROWS: { kind: keyof Usage["limits"]; label: string }[] = [
  { kind: "question", label: "Questions" },
  { kind: "quiz", label: "Quizzes" },
  { kind: "upload", label: "Uploads" },
];

function UsageMeter({ usage }: { usage: Usage }) {
  const rows = USAGE_ROWS.filter((r) => usage.limits[r.kind].limit > 0);
  if (rows.length === 0) return null;
  const resets = new Date(usage.resets_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  return (
    <div className="usage" title={`Daily limits reset at ${resets} your time`}>
      <div className="usage-title">Today</div>
      {rows.map(({ kind, label }) => {
        const { used, limit } = usage.limits[kind];
        const ratio = Math.min(used / limit, 1);
        return (
          <div key={kind} className="usage-row">
            <span>{label}</span>
            <span className={ratio >= 1 ? "usage-full" : ratio >= 0.8 ? "usage-low" : ""}>
              {used}/{limit}
            </span>
            <div className="usage-bar">
              <div style={{ width: `${ratio * 100}%` }} className={ratio >= 1 ? "full" : ratio >= 0.8 ? "low" : ""} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

interface Props {
  sections: Section[];
  activeId: string | null;
  user: User;
  usage: Usage | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
  onLogout: () => void;
}

export default function Sidebar({ sections, activeId, user, usage, onSelect, onNewChat, onDelete, onLogout }: Props) {
  return (
    <aside className="sidebar">
      <button className="new-chat" onClick={onNewChat}>
        <span>＋</span> New chat
      </button>
      <div className="sidebar-label">Chats</div>
      <nav className="section-list">
        {sections.length === 0 && <div className="muted small pad">No chats yet</div>}
        {sections.map((s) => (
          <div
            key={s.id}
            className={`section-item ${s.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(s.id)}
            title={s.title ?? "New chat"}
          >
            <span className="section-title">{s.title ?? "New chat"}</span>
            <span className={`count ${s.status === "limit_reached" ? "full" : ""}`}>
              {s.question_count}/{s.max_questions}
            </span>
            <button
              className="delete"
              aria-label="Delete chat"
              onClick={(e) => {
                e.stopPropagation();
                if (confirm("Delete this chat and its history?")) onDelete(s.id);
              }}
            >
              ×
            </button>
          </div>
        ))}
      </nav>
      {usage && <UsageMeter usage={usage} />}
      <div className="sidebar-footer">
        <span className="small muted ellipsis">{user.email}</span>
        <button className="link small" onClick={onLogout}>
          Sign out
        </button>
      </div>
    </aside>
  );
}
