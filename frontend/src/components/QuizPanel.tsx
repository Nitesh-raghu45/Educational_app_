import { FormEvent, useState } from "react";
import { api, describeError } from "../api";
import type { Attempt, Quiz } from "../types";
import { CloseIcon } from "./icons";

export default function QuizPanel({ sectionId, onClose }: { sectionId: string; onClose: () => void }) {
  const [count, setCount] = useState(5);
  const [topic, setTopic] = useState("");
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function generate(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const q = await api.createQuiz(sectionId, count, topic.trim());
      setQuiz(q);
      setAnswers(q.questions.map(() => null));
      setAttempt(null);
    } catch (err) {
      setError(describeError(err, "Couldn't create a quiz."));
    } finally {
      setBusy(false);
    }
  }

  async function submit() {
    if (!quiz) return;
    setBusy(true);
    try {
      setAttempt(await api.submitAttempt(quiz.id, answers));
    } catch (err) {
      setError(describeError(err, "Couldn't submit answers."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Quiz">
        <div className="panel-header">
          <h3>{attempt ? `Score: ${attempt.score} / ${attempt.total}` : "Quiz from your documents"}</h3>
          <button className="icon" onClick={onClose} aria-label="Close quiz">
            <CloseIcon />
          </button>
        </div>

        {!quiz && (
          <form className="quiz-form" onSubmit={generate}>
            <label>
              Topic (optional)
              <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="e.g. photosynthesis" maxLength={200} />
            </label>
            <label>
              Number of questions
              <input type="number" min={1} max={15} value={count} onChange={(e) => setCount(Number(e.target.value))} />
            </label>
            {error && <div className="error">{error}</div>}
            <button className="primary" disabled={busy}>
              {busy ? "Generating…" : "Generate quiz"}
            </button>
          </form>
        )}

        {quiz && (
          <div className="quiz">
            {quiz.questions.map((q, qi) => {
              const result = attempt?.results[qi];
              return (
                <fieldset key={qi} className="quiz-question">
                  <legend>
                    {qi + 1}. {q.question}
                  </legend>
                  {q.options.map((option, oi) => {
                    const cls = result
                      ? oi === result.correct_index
                        ? "correct"
                        : oi === result.selected
                          ? "wrong"
                          : ""
                      : "";
                    return (
                      <label key={oi} className={`option ${cls}`}>
                        <input
                          type="radio"
                          name={`q${qi}`}
                          checked={answers[qi] === oi}
                          disabled={!!attempt}
                          onChange={() => setAnswers((a) => a.map((v, i) => (i === qi ? oi : v)))}
                        />
                        {option}
                      </label>
                    );
                  })}
                  {result && (
                    <div className="explanation small">
                      {result.explanation}
                      {result.citation && (
                        <div className="muted">
                          Source: {result.citation.filename}
                          {result.citation.page ? `, p.${result.citation.page}` : ""}
                        </div>
                      )}
                    </div>
                  )}
                </fieldset>
              );
            })}
            {error && <div className="error">{error}</div>}
            <div className="quiz-actions">
              {!attempt ? (
                <button className="primary" onClick={submit} disabled={busy || answers.some((a) => a === null)}>
                  Submit answers
                </button>
              ) : (
                <button className="primary" onClick={() => setQuiz(null)}>
                  New quiz
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
