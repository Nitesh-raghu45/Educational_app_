import { useCallback, useEffect, useState } from "react";
import { api, describeError, getToken, setToken, setUnauthorizedHandler } from "./api";
import AuthScreen from "./components/AuthScreen";
import ChatView from "./components/ChatView";
import Sidebar from "./components/Sidebar";
import { MenuIcon } from "./components/icons";
import type { Section, Usage, User } from "./types";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(!!getToken());
  const [sections, setSections] = useState<Section[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    setSections([]);
    setActiveId(null);
  }, []);

  useEffect(() => setUnauthorizedHandler(logout), [logout]);

  // Esc closes the sidebar drawer on small screens.
  useEffect(() => {
    if (!sidebarOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setSidebarOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sidebarOpen]);

  useEffect(() => {
    if (!getToken()) return;
    api
      .me()
      .then(setUser)
      .catch(() => logout())
      .finally(() => setChecking(false));
  }, [logout]);

  useEffect(() => {
    if (!user) return;
    api.listSections().then((list) => {
      setSections(list);
      setActiveId((current) => current ?? list[0]?.id ?? null);
    });
  }, [user]);

  // Refresh today's usage whenever chats change (a new chat, or a question updating a chat).
  useEffect(() => {
    if (!user) return;
    api.usage().then(setUsage).catch(() => {});
  }, [user, sections]);

  const upsertSection = useCallback((section: Section) => {
    setSections((list) =>
      [section, ...list.filter((s) => s.id !== section.id)].sort((a, b) => b.updated_at.localeCompare(a.updated_at)),
    );
  }, []);

  const newChat = useCallback(async () => {
    try {
      const section = await api.createSection();
      setNotice(null);
      setSections((list) => [section, ...list]);
      setActiveId(section.id);
      setSidebarOpen(false);
    } catch (err) {
      setNotice(describeError(err, "Couldn't start a new chat."));
    }
  }, []);

  const deleteChat = useCallback(
    async (id: string) => {
      await api.deleteSection(id);
      setSections((list) => {
        const rest = list.filter((s) => s.id !== id);
        if (activeId === id) setActiveId(rest[0]?.id ?? null);
        return rest;
      });
    },
    [activeId],
  );

  if (checking) return <div className="center muted">Loading…</div>;
  if (!user)
    return (
      <AuthScreen
        onAuthenticated={(token, u) => {
          setToken(token);
          setUser(u);
        }}
      />
    );

  return (
    <div className={`layout ${sidebarOpen ? "sidebar-open" : ""}`}>
      <Sidebar
        sections={sections}
        activeId={activeId}
        user={user}
        usage={usage}
        onSelect={(id) => {
          setActiveId(id);
          setSidebarOpen(false);
        }}
        onNewChat={newChat}
        onDelete={deleteChat}
        onLogout={logout}
      />
      <div className="scrim" onClick={() => setSidebarOpen(false)} />
      <main className="main">
        <button className="menu-button" aria-label="Open sidebar" onClick={() => setSidebarOpen(true)}>
          <MenuIcon />
        </button>
        {notice && (
          <div className="notice" role="alert">
            {notice}
            <button className="icon" aria-label="Dismiss" onClick={() => setNotice(null)}>
              ×
            </button>
          </div>
        )}
        {activeId ? (
          <ChatView key={activeId} sectionId={activeId} onSectionChange={upsertSection} onNewChat={newChat} />
        ) : (
          <div className="empty-state">
            <h1>Study Assistant</h1>
            <p className="muted">Upload your notes and ask questions answered only from your documents.</p>
            <button className="primary" onClick={newChat}>
              Start a new chat
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
