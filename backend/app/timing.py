"""Where the seconds go between a question and its answer, logged as one line per answer.

The timer for the current answer lives in a context variable, so code deep in the pipeline
(embedding, Pinecone, keyword search) can record its step without passing the timer around.
Tasks and threads started from the answer (asyncio.gather, asyncio.to_thread) copy the context,
so they record into the same timer.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

_current: ContextVar["AnswerTimer | None"] = ContextVar("answer_timer", default=None)


class AnswerTimer:
    def __init__(self, started: float | None = None):
        self.started = started if started is not None else time.perf_counter()
        self.steps: dict[str, float] = {}
        self.facts: dict[str, str] = {}
        self.first_word: float | None = None

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        began = time.perf_counter()
        try:
            yield
        finally:
            self.steps[name] = self.steps.get(name, 0.0) + time.perf_counter() - began

    def mark_first_word(self) -> None:
        if self.first_word is None:
            self.first_word = time.perf_counter() - self.started

    def note(self, name: str, value: object) -> None:
        self.facts[name] = str(value)

    def report(self) -> str:
        total = time.perf_counter() - self.started
        first = f"{self.first_word:.1f}s" if self.first_word is not None else "none"
        steps = " | ".join(f"{name} {seconds:.2f}s" for name, seconds in self.steps.items())
        facts = ", ".join(f"{name}={value}" for name, value in self.facts.items())
        return f"total {total:.1f}s, first word at {first} :: {steps} :: {facts}"


def start(started: float | None = None) -> AnswerTimer:
    timer = AnswerTimer(started)
    _current.set(timer)
    return timer


def current() -> AnswerTimer | None:
    return _current.get()


def step(name: str):
    """Time a step of the current answer (does nothing outside one)."""
    timer = _current.get()
    return timer.step(name) if timer else nullcontext()


def now() -> float:
    """A FastAPI dependency: listed first, it records when the request started being handled."""
    return time.perf_counter()
