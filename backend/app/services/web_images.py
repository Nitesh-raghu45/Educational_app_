"""Pictures from the web, shown in answers as illustrations (never as evidence).

Sources: Wikipedia article images (free, no key, openly licensed) and, when SERPER_API_KEY is
set, Google Images through Serper.dev. The search runs alongside document retrieval and is cut
off after `web_image_timeout` seconds, so it never delays an answer; results are cached so a
repeated topic costs no search at all.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlparse

import httpx
from redis.exceptions import RedisError

from ..config import get_settings
from ..ratelimit import get_redis

logger = logging.getLogger(__name__)

USER_AGENT = "StudyAssistant/1.0 (educational app; https://github.com/Nitesh-raghu45/Educational_app_)"
MAX_CANDIDATES = 4
_CACHE_PREFIX = "webimg:v1:"
_MEMORY_CACHE_SIZE = 500


@dataclass
class WebImage:
    title: str
    # Shown in the answer (a ~800 px version where the provider has one).
    image_url: str
    # The page the picture comes from (for Wikipedia, its file page with author and licence).
    page_url: str
    # Domain shown under the image, e.g. "wikipedia.org".
    source: str
    provider: str  # "wikipedia" or "google"
    description: str = ""
    width: int | None = None
    height: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageSearch(Protocol):
    async def search(self, query: str, *, use_google: bool) -> list[WebImage]: ...


# --- When to search -------------------------------------------------------------------------

# The student asked for a picture ("show me an image of…", "photo dikhao", "चित्र").
_EXPLICIT_RE = re.compile(
    r"\b(?:images?|pictures?|pics?|photos?|photographs?|illustrations?|show\s+me|looks?\s+like|dikhao|dikhaiye"
    r"|tasveer|tasvir|chitra|photu)\b|चित्र|तस्वीर|फ़ोटो|फोटो|दिखाओ|दिखाइए",
    re.IGNORECASE,
)
# Topics where a picture usually helps even unasked: physical things and how they look.
_HELPFUL_RE = re.compile(
    r"\b(?:structure|anatomy|parts?\s+of|label(?:l?ed)?|shape|appearance|organs?|cells?|tissues?|skeleton|bones?"
    r"|muscles?|heart|brain|lungs?|kidneys?|eye|ear|neuron|circuit|apparatus|instrument|machine|engine|device"
    r"|monument|temple|fort|painting|sculpture|map|continent|mountain|river|volcano|planet|galaxy|solar\s+system"
    r"|species|animals?|birds?|insects?|plants?|flowers?|leaf|leaves|crystals?|minerals?|rocks?"
    r"|who\s+(?:was|is))\b",
    re.IGNORECASE,
)


def image_intent(question: str) -> str | None:
    """"explicit" when the student asked for a picture, "helpful" when one would likely help, else None."""
    if _EXPLICIT_RE.search(question):
        return "explicit"
    if _HELPFUL_RE.search(question):
        return "helpful"
    return None


_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with", "about", "from", "by", "at", "as",
    "is", "are", "was", "were", "be", "do", "does", "did", "what", "which", "who", "whom", "how", "why",
    "when", "where", "can", "could", "would", "will", "should", "please", "me", "us", "my", "i", "you",
    "it", "its", "this", "that", "these", "those", "some", "any", "give", "show", "tell", "explain",
    "describe", "draw", "find", "get", "need", "want", "see", "look", "looks", "like", "also",
    "image", "images", "picture", "pictures", "pic", "pics", "photo", "photos", "photograph", "photographs",
    "illustration", "illustrations", "diagram", "diagrams", "labelled", "labeled", "kya", "hai", "ka", "ki",
    "ke", "ko", "mujhe", "dikhao", "dikhaiye", "batao", "samjhao", "tasveer", "tasvir", "chitra", "photu",
}


def search_query(text: str) -> str:
    """Keep a question's topic words: "show me a picture of the human heart" -> "human heart"."""
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text.lower())
    kept = [w for w in words if w not in _STOPWORDS and not w.isdigit()]
    return " ".join(kept[:8])


# --- Providers ------------------------------------------------------------------------------


def _https(url: Any) -> str | None:
    """Only https image URLs: the app is served over https and must not load mixed content."""
    return url if isinstance(url, str) and url.startswith("https://") and len(url) <= 2000 else None


def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.removeprefix("www.").removeprefix("en.").removeprefix("upload.")


async def _wikipedia(client: httpx.AsyncClient, query: str) -> list[WebImage]:
    """Lead images of the best-matching Wikipedia articles: curated, relevant and freely licensed."""
    response = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": "4",
            "gsrnamespace": "0",
            "prop": "pageimages|description",
            "piprop": "thumbnail|name",
            "pithumbsize": "800",
            "pilicense": "free",
        },
    )
    response.raise_for_status()
    pages = sorted(response.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
    images = []
    for page in pages:
        thumb = page.get("thumbnail") or {}
        url = _https(thumb.get("source"))
        name = page.get("pageimage")
        if not url or not name:
            continue
        images.append(
            WebImage(
                title=page.get("title", ""),
                image_url=url,
                page_url=f"https://en.wikipedia.org/wiki/File:{quote(name)}",
                source="wikipedia.org",
                provider="wikipedia",
                description=page.get("description", ""),
                width=thumb.get("width"),
                height=thumb.get("height"),
            )
        )
    return images[:2]


# Serper's images endpoint may not accept Google's SafeSearch field; if it rejects it once, stop sending it.
_serper_safe_supported = True


async def _google(client: httpx.AsyncClient, query: str, api_key: str) -> list[WebImage]:
    """Google Images results through Serper.dev (Google's own image API is closed to new users)."""
    global _serper_safe_supported
    body: dict[str, Any] = {"q": query, "num": 10}
    if _serper_safe_supported:
        body["safe"] = "active"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    response = await client.post("https://google.serper.dev/images", json=body, headers=headers)
    if response.status_code == 400 and "safe" in body:
        _serper_safe_supported = False
        body.pop("safe")
        response = await client.post("https://google.serper.dev/images", json=body, headers=headers)
    response.raise_for_status()
    images = []
    for item in response.json().get("images", []):
        url, page = _https(item.get("imageUrl")), _https(item.get("link"))
        if not url or not page:
            continue
        images.append(
            WebImage(
                title=str(item.get("title", ""))[:200],
                image_url=url,
                page_url=page,
                source=str(item.get("domain") or _domain(page)),
                provider="google",
                width=item.get("imageWidth"),
                height=item.get("imageHeight"),
            )
        )
        if len(images) == 2:
            break
    return images


class WebImageSearch:
    def __init__(self) -> None:
        # One long-lived client: reusing its open connections saves a TLS handshake (often
        # several hundred ms) on every search after the first.
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=5.0,
                limits=httpx.Limits(keepalive_expiry=300),
            )
        return self._client

    async def warm_up(self) -> None:
        """Open the connection to Wikipedia at startup, so the first search is as fast as later ones."""
        try:
            await self._http().head("https://en.wikipedia.org/w/api.php")
        except httpx.HTTPError:
            logger.info("Couldn't reach Wikipedia at startup; web pictures will connect on first use")

    async def search(self, query: str, *, use_google: bool) -> list[WebImage]:
        settings = get_settings()
        google = use_google and bool(settings.serper_api_key)
        client = self._http()
        calls = [_wikipedia(client, query)]
        if google:
            calls.append(_google(client, query, settings.serper_api_key))
        results = await asyncio.gather(*calls, return_exceptions=True)
        images: list[WebImage] = []
        for name, result in zip(["wikipedia", "google"], results):
            if isinstance(result, BaseException):
                logger.warning("Web image search (%s) failed: %s", name, result)
            else:
                images.extend(result)
        seen: set[str] = set()
        unique = [i for i in images if not (i.image_url in seen or seen.add(i.image_url))]
        return unique[:MAX_CANDIDATES]


_search: ImageSearch | None = None


def image_search() -> ImageSearch:
    global _search
    if _search is None:
        _search = WebImageSearch()
    return _search


async def warm_up() -> None:
    search = image_search()
    if get_settings().web_images_enabled and isinstance(search, WebImageSearch):
        await search.warm_up()


def override(search: ImageSearch | None) -> None:
    global _search
    _search = search


# --- Cache ----------------------------------------------------------------------------------

_memory: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _cache_key(query: str, use_google: bool) -> str:
    digest = hashlib.sha256(f"{use_google}:{query}".encode()).hexdigest()[:32]
    return _CACHE_PREFIX + digest


async def _cache_get(key: str) -> list[WebImage] | None:
    client = get_redis()
    raw: Any = None
    if client is not None:
        try:
            raw = await client.get(key)
        except RedisError:
            logger.warning("Web image cache unavailable", exc_info=True)
        if raw is not None:
            return [WebImage(**item) for item in json.loads(raw)]
    entry = _memory.get(key)
    if entry and entry[0] > time.time():
        return [WebImage(**item) for item in entry[1]]
    return None


async def _cache_set(key: str, images: list[WebImage]) -> None:
    ttl = get_settings().web_image_cache_seconds
    items = [i.as_dict() for i in images]
    client = get_redis()
    if client is not None:
        try:
            await client.set(key, json.dumps(items), ex=ttl)
            return
        except RedisError:
            logger.warning("Web image cache unavailable", exc_info=True)
    if len(_memory) >= _MEMORY_CACHE_SIZE:
        _memory.pop(next(iter(_memory)))
    _memory[key] = (time.time() + ttl, items)


async def find_images(query: str, *, use_google: bool) -> list[WebImage]:
    """Cached search. Never raises: a failed search means an answer without web pictures."""
    if not query:
        return []
    key = _cache_key(query, use_google)
    try:
        cached = await _cache_get(key)
        if cached is not None:
            return cached
        images = await image_search().search(query, use_google=use_google)
        await _cache_set(key, images)
        return images
    except Exception:
        logger.warning("Web image search failed for %r", query, exc_info=True)
        return []


# --- Use in answers -------------------------------------------------------------------------


def start_search(question: str, query: str) -> "asyncio.Task[list[WebImage]] | None":
    """Begin a search in the background when a picture may help; None when it won't be needed."""
    if not get_settings().web_images_enabled:
        return None
    intent = image_intent(question) or image_intent(query)
    if intent is None:
        return None
    topic = search_query(query) or search_query(question)
    # Google costs money per search, so it is used only when the student asked for a picture.
    task = asyncio.create_task(find_images(topic, use_google=intent == "explicit"))
    # Keep a reference: a search that outlives its answer must not be garbage-collected mid-way.
    _running.add(task)
    task.add_done_callback(_running.discard)
    return task


_running: set[asyncio.Task] = set()


async def collect(task: "asyncio.Task[list[WebImage]] | None", started: float) -> list[WebImage]:
    """Results if the search finished within the time limit (counted from `started`), else none.

    A late search keeps running in the background and fills the cache for next time.
    """
    if task is None:
        return []
    remaining = get_settings().web_image_timeout - (time.monotonic() - started)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=max(remaining, 0.0))
    except asyncio.TimeoutError:
        logger.info("Web image search took too long; answering without it")
        return []


_WEB_TAG_RE = re.compile(r"<\s*/?\s*(?:source|image)\b", re.IGNORECASE)


def _clean(text: str) -> str:
    return _WEB_TAG_RE.sub("‹", " ".join(text.split())).replace('"', "'")[:200]


def format_web_images(images: list[WebImage]) -> str:
    lines = [
        f'<image id="{n}" title="{_clean(i.title)}" from="{_clean(i.source)}"'
        + (f' about="{_clean(i.description)}"' if i.description else "")
        + " />"
        for n, i in enumerate(images, 1)
    ]
    return "\n".join(lines)
