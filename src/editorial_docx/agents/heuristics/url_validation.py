from __future__ import annotations

import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ...models import AgentComment
from ...review_patterns import _ref_block_type

_URL_RE = re.compile(r"https?://[^\s<>]+", flags=re.IGNORECASE)


def _enabled() -> bool:
    return os.getenv("REFERENCE_URL_VALIDATION", "false").strip().casefold() in {"1", "true", "yes"}


def _url_status(url: str, timeout: float = 8.0) -> int | None:
    """Returns a definitive broken-link status, never treating connection errors as broken links."""
    request = Request(url, headers={"User-Agent": "IPEAREV/0.2 link-validation", "Range": "bytes=0-0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200))
    except HTTPError as exc:
        return exc.code
    except (URLError, TimeoutError, OSError, ValueError):
        return None


def heuristic_broken_url_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    """Flags only definitive 404/410 reference URLs when external checking is enabled."""
    if not _enabled():
        return []
    comments: list[AgentComment] = []
    for idx in batch_indexes:
        if not (0 <= idx < len(chunks)) or idx >= len(refs) or _ref_block_type(refs[idx]) != "reference_entry":
            continue
        text = chunks[idx] or ""
        for raw_url in _URL_RE.findall(text):
            url = raw_url.rstrip(".,;:)")
            status = _url_status(url)
            if status not in {404, 410}:
                continue
            comments.append(
                AgentComment(
                    agent="referencias",
                    category="reference_link",
                    message=f"O endereço da referência retornou HTTP {status} e não foi localizado.",
                    paragraph_index=idx,
                    issue_excerpt=url,
                    suggested_fix="Solicitar ao autor um URL ou DOI válido para esta referência.",
                    action_type="author_confirmation",
                )
            )
    return comments
