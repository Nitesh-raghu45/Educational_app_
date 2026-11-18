"""Decide, before any search, what a message needs: rules only, so it adds no waiting.

- "chat": greetings, thanks, "ok", "bye", "what can you do?" → a fixed reply. No search, no model.
- "rework": redo the previous answer (translate, shorten, simplify, points or a table, again)
  → the model with that answer only. No rewrite, no search.
- "documents": everything else → the normal search-and-answer flow.

When unsure it says "documents": searching for nothing costs about a second, while skipping a
search that was needed could let the model answer from its own knowledge instead of the chapter.
Every decision is logged ("Route: …") so the rules can be tuned from real messages later.
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Route = Literal["chat", "rework", "documents"]


@dataclass(frozen=True)
class Decision:
    route: Route
    reason: str
    reply: str = ""  # the fixed reply, for "chat"


# ---------- chat ----------

# Each kind: phrases that may make up the WHOLE message, in any order ("ok thanks sir").
_CHAT_PHRASES: dict[str, str] = {
    "thanks": (
        r"thank\s*(?:you|u|yu)|thanks?|thanx|thnx|thnks|thx|tysm|ty|thanku|thankyou|dhanyavaa?d|dhanyawaa?d|shukriya"
        r"|धन्यवाद|शुक्रिया"
    ),
    "bye": r"bye+|good\s*bye|see\s+(?:you|ya|u)|tata|alvida|good\s*night|अलविदा",
    "greeting": (
        r"h+i+|hey+|hello+|helo+|hlo+|hii+|yo|namaste|namaskar|good\s+(?:morning|afternoon|evening|day)"
        r"|नमस्ते|नमस्कार"
    ),
    "ack": (
        r"o+k+|okay+|okey|k+|alright|all\s+right|fine|cool|nice|great|awesome|perfect|good|wow|hmm+|got\s+it"
        r"|understood|i\s+understood|i\s+understand|i\s+get\s+it|clear|samajh\s+(?:gaya|gayi|gya|gyi|aa\s+gaya|aaya)"
        r"|th[ei]+k\s+hai|achh?a|acch?a|badhiya|ठीक\s+है|समझ\s+गया|समझ\s+गई|अच्छा"
    ),
    "about": (
        r"who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|how\s+do\s+you\s+work|how\s+(?:to|do\s+i)\s+use\s+(?:you|this|this\s+app)"
        r"|help|what\s+do\s+you\s+do|tum\s+kaun\s+ho|aap\s+kaun\s+ho"
    ),
}
# Words that may go with them: "thank you so much sir", "hi there", "ok bhai".
_CHAT_FILLER = (
    r"so|much|very|a\s+lot|lot|a|again|there|all|everyone|sir|maam|ma'am|mam|madam|miss|bhai|bro|dear|buddy|ji|didi"
    r"|guys|friend|for|the|help|your|answer|explanation|it|it's|its|that|this|now|and|really|too|you|u|me|haan|ha"
)
# Messages are matched word by word after _words(): a phrase must end where a word ends
# ((?=\s|$), not \b, which fails after Hindi vowel signs).
_END = r"(?=\s|$)"
_CHAT_KIND_RES = {kind: re.compile(rf"^(?:{pattern}){_END}", re.IGNORECASE) for kind, pattern in _CHAT_PHRASES.items()}
_FILLER_RE = re.compile(rf"^(?:{_CHAT_FILLER}){_END}", re.IGNORECASE)
# Emoji, punctuation and the like between words (Devanagari, with its vowel signs, is kept).
_NOISE_RE = re.compile(r"[^\w\s'ऀ-ॿ]+", re.UNICODE)


def _words(text: str) -> str:
    return " ".join(_NOISE_RE.sub(" ", text).split())

# Most specific first: "ok thanks" is thanks, "hi, what can you do" is about.
_CHAT_PRIORITY = ("about", "thanks", "bye", "greeting", "ack")
CHAT_REPLIES = {
    "greeting": "Hi! 👋 Ask me anything about your chapter — a definition, a topic to explain, or its important questions.",
    "thanks": "You're welcome! 😊 Ask your next question whenever you're ready.",
    "ack": "Great! Ask your next question whenever you're ready.",
    "bye": "Bye! Good luck with your studies. 📚",
    "about": (
        "I'm your study assistant. I answer from the chapter or files attached to this chat: I can explain "
        "topics in simple words, list the chapter's important questions, show its figures, draw diagrams, "
        "and quiz you (use **Quiz me** at the top). Ask in English, Hindi or Hinglish."
    ),
}
CHAT_REPLIES_HINDI = {
    "greeting": "नमस्ते! 👋 अपने अध्याय के बारे में कुछ भी पूछिए।",
    "thanks": "आपका स्वागत है! 😊 अगला सवाल जब चाहें पूछिए।",
    "ack": "बढ़िया! अगला सवाल जब चाहें पूछिए।",
    "bye": "अलविदा! पढ़ाई के लिए शुभकामनाएँ। 📚",
    "about": CHAT_REPLIES["about"],
}
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Saved answers that are small talk, not something to translate or shorten.
CHAT_REPLY_TEXTS = frozenset(CHAT_REPLIES.values()) | frozenset(CHAT_REPLIES_HINDI.values())


def _chat_kind(question: str) -> str | None:
    """The kind of small talk, when the WHOLE message is small talk; otherwise None."""
    text = _words(question)
    if not text or len(text.split()) > 10:
        return None
    kinds: set[str] = set()
    while text:
        for kind, pattern in _CHAT_KIND_RES.items():
            if m := pattern.match(text):
                kinds.add(kind)
                break
        else:
            if not (m := _FILLER_RE.match(text)):
                return None  # a word that isn't small talk: a real question
        text = text[m.end():].lstrip()
    return next((k for k in _CHAT_PRIORITY if k in kinds), None)


# ---------- rework the previous answer ----------

# What the student wants done to the previous answer. One of these must be present.
_REWORK_ACTIONS: dict[str, str] = {
    "language": (
        r"translate|hindi|english|hinglish|marathi|gujarati|bengali|bangla|tamil|telugu|kannada|punjabi|urdu"
        r"|हिंदी|हिन्दी|अंग्रेजी"
    ),
    "shorter": (
        r"shorter|short|shorten|brief|briefly|concise|concisely"
        r"|in\s+few\s+words|chhota|chota|kam\s+shabdon|tl;?dr"
    ),
    "simpler": (
        r"simpler|simple|simply|easier|easy|easily|layman|plain|basic|kid|child|aasan|asaan|saral|aasaan"
        r"|सरल|आसान"
    ),
    "format": (
        r"points?|point\s*wise|pointwise|bullets?|bullet\s+points|table|tabular|tabulate|paragraph|numbered|list"
    ),
    "repeat": r"again|repeat|once\s+more|one\s+more\s+time|phir\s+se|fir\s+se|dobara|dubara|rephrase|reword|differently|other\s+way",
}
_REWORK_ACTION_RES = {kind: re.compile(rf"^(?:{p}){_END}", re.IGNORECASE) for kind, p in _REWORK_ACTIONS.items()}
# Words that carry no topic: "can you please explain it in simple words".
_REWORK_FILLER_RE = re.compile(
    r"^(?:can|could|would|will|you|u|please|pls|plz|kindly|just|now|ok|okay|also|and|then|so|but"
    r"|explain|say|said|write|give|make|put|rewrite|tell|show|convert|turn|change|do|it|it's|its|this|that|the|a|an"
    r"|answer|reply|above|previous|last|same|whole|your|me|my|i|we|in|into|to|as|of|for|with|form|format|way|language"
    r"|words?|terms|bit|little|lil|more|much|very|really|style|version|one|sentences?|lines?|like|i'm|im|am|10|ten"
    r"|year|years|old|want|need|it's|please|again|batao|bataiye|batayiye|samjhao|samjhaiye|samjha|likho|likhiye|karo|kar"
    r"|karke|kariye|do|dijiye|isko|ise|isse|use|usko|yeh|ye|ise|wo|woh|me|mein|main|mai|ko|mujhe|hamein|thoda|zara"
    r"|bhasha|language|mein|का|की|के|में|को|इसे|इसको|बताओ|समझाओ|लिखो|भाषा)(?=\s|$)",
    re.IGNORECASE,
)
# Asking for MORE than the answer had needs the documents: "explain in more detail", "with examples".
_NEEDS_MORE_RE = re.compile(
    r"\b(?:elaborate|detail|details|detailed|expand|example|examples|deeper|deep|longer|long|full|complete"
    r"|why|how|what|which|where|when|who|vistar|vistaar)\b",
    re.IGNORECASE,
)


def _rework_kind(question: str) -> str | None:
    """What to do to the previous answer, when the message asks only that; otherwise None."""
    text = _words(question)
    if not text or len(text.split()) > 14 or _NEEDS_MORE_RE.search(text):
        return None
    kinds: list[str] = []
    while text:
        for kind, pattern in _REWORK_ACTION_RES.items():
            if m := pattern.match(text):
                kinds.append(kind)
                break
        else:
            if not (m := _REWORK_FILLER_RE.match(text)):
                return None  # a topic word ("explain tangents in hindi"): search for it
        text = text[m.end():].lstrip()
    return kinds[0] if kinds else None


# ---------- the decision ----------


def route(question: str) -> Decision:
    """Route a message by its text alone. "rework" still needs a previous answer to work on:
    the caller checks that and falls back to "documents" if there is none."""
    if kind := _chat_kind(question):
        replies = CHAT_REPLIES_HINDI if _DEVANAGARI_RE.search(question) else CHAT_REPLIES
        return Decision("chat", kind, replies[kind])
    if kind := _rework_kind(question):
        return Decision("rework", kind)
    return Decision("documents", "default")


def log_decision(decision: Decision, question: str) -> None:
    """One line per message, for tuning the rules later: grep the logs for "Route:"."""
    shown = " ".join(question.split())
    shown = shown if len(shown) <= 200 else shown[:197] + "..."
    logger.info("Route: route=%s reason=%s question=%r", decision.route, decision.reason, shown)
