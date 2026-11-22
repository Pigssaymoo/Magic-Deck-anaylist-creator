"""Rate-limited HTTP with disk caching, backoff, and robots.txt respect.

All outbound traffic for the three recommendation sources plus MTG Singles NZ
goes through this class so rate limits and cache TTLs are enforced in one place.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from urllib import robotparser

import requests

from .config import Config

#: Scryfall (and other well-behaved APIs) reject default library user agents,
#: so every request — including direct file downloads — carries this one.
USER_AGENT = "cmdr-deckbuilder/1.0 (personal deck-building tool)"


class SourceBlocked(RuntimeError):
    """A source refused the request (HTTP error, robots.txt, or retries exhausted)."""


@dataclass
class RateLimiter:
    """Enforces a minimum interval between requests per host."""

    intervals: dict[str, float]
    last: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.last is None:
            self.last = {}

    def wait(self, host: str) -> None:
        interval = self.intervals.get(host, self.intervals.get("*", 1.0))
        prev = self.last.get(host)
        if prev is not None:
            elapsed = time.monotonic() - prev
            if elapsed < interval:
                time.sleep(interval - elapsed)
        self.last[host] = time.monotonic()


class Http:
    """requests wrapper: per-host rate limit, TTL disk cache, backoff, robots check."""

    def __init__(self, cfg: Config, user_agent: str = USER_AGENT) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.cache_dir = Path(cfg.http.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter({"*": cfg.http.default_min_interval_sec})
        self._robots_ok: dict[str, bool] = {}

    # ---------------------------------------------------------------- robots
    def robots_allows(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        if host in self._robots_ok:
            return self._robots_ok[host]
        ok = True
        robots_url = f"{parsed.scheme}://{host}/robots.txt"
        try:
            r = self.session.get(robots_url, timeout=20)
            if r.status_code == 200 and r.text.strip():
                rp = robotparser.RobotFileParser()
                rp.parse(r.text.splitlines())
                ok = rp.can_fetch(self.session.headers["User-Agent"], url)
            # 403/404/empty robots.txt -> no explicit restriction
        except requests.RequestException:
            ok = True
        self._robots_ok[host] = ok
        return ok

    # ------------------------------------------------------------------ cache
    def _key(self, method: str, url: str, json_body: dict | None, params: dict | None) -> str:
        h = hashlib.sha256()
        h.update(f"{method.upper()}|{url}|".encode("utf-8"))
        if json_body is not None:
            h.update(json.dumps(json_body, sort_keys=True).encode("utf-8"))
        if params is not None:
            h.update(json.dumps(params, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def _read_cache(self, key: str, ttl: float) -> bytes | None:
        meta_file = self.cache_dir / f"{key}.meta.json"
        body_file = self.cache_dir / f"{key}.body"
        if meta_file.exists() and body_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if time.time() - float(meta.get("fetched_at", 0)) < ttl:
                    return body_file.read_bytes()
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _write_cache(self, key: str, body: bytes) -> None:
        try:
            (self.cache_dir / f"{key}.body").write_bytes(body)
            (self.cache_dir / f"{key}.meta.json").write_text(
                json.dumps({"fetched_at": time.time()}), encoding="utf-8"
            )
        except OSError:
            pass  # cache write failure is non-fatal

    # ---------------------------------------------------------------- request
    def request(
        self,
        method: str,
        url: str,
        *,
        ttl: float,
        headers: dict[str, str] | None = None,
        json_body: dict | None = None,
        params: dict | None = None,
        force: bool = False,
    ) -> bytes:
        """Perform an HTTP request with caching, rate limiting, and backoff."""
        key = self._key(method, url, json_body, params)
        if not force:
            cached = self._read_cache(key, ttl)
            if cached is not None:
                return cached

        host = urllib.parse.urlparse(url).netloc
        if not self.robots_allows(url):
            raise SourceBlocked(f"robots.txt disallows access to {url}")

        backoff = 2.0
        max_retries = self.cfg.http.max_retries
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            self.limiter.wait(host)
            try:
                r = self.session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    params=params,
                    timeout=self.cfg.http.timeout_sec,
                )
            except requests.RequestException as exc:  # network-level failure
                last_error = exc
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                raise SourceBlocked(f"network error for {url}: {exc}") from exc

            if r.status_code in (429, 500, 502, 503, 529):
                last_error = SourceBlocked(f"{method} {url} -> HTTP {r.status_code}")
                if attempt < max_retries:
                    retry_after = r.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else backoff
                    time.sleep(min(delay, 30.0))
                    backoff = min(backoff * 2, 30.0)
                    continue
                raise last_error
            if r.status_code >= 400:
                raise SourceBlocked(f"{method} {url} -> HTTP {r.status_code}: {r.text[:200]!r}")

            body = r.content
            self._write_cache(key, body)
            return body

        raise SourceBlocked(f"exhausted retries for {url}: {last_error}")

    def cached_body(self, url: str, *, ttl: float, params: dict | None = None) -> bytes | None:
        """Fresh on-disk cache body for a GET (url, params), or None.

        Probe-free: never touches the network and never rate-limits. A cache
        miss (or an expired entry) simply means "no data" -- callers use this
        for reads that must not burn rate-limited lookups, such as the budget
        floor precheck deciding whether a build is even possible.
        """
        return self._read_cache(self._key("GET", url, None, params), ttl)

    def _purge(self, key: str) -> None:
        for suffix in (".meta.json", ".body"):
            try:
                (self.cache_dir / f"{key}{suffix}").unlink()
            except OSError:
                pass

    def _get_json(self, method: str, url: str, *, ttl: float, **kw) -> dict:
        """JSON GET/POST where a non-JSON body is a source failure, not a crash.

        A 200 response that is not JSON (bot-block page, error text, empty
        body) is degraded to ``SourceBlocked`` -- the same class every caller
        already handles for unreachable sources -- and is never kept: a
        poisoned cache entry is purged and re-fetched once, and a fresh bad
        body is purged before the error is raised.
        """
        force = bool(kw.pop("force", False))
        key = self._key(method, url, kw.get("json_body"), kw.get("params"))
        if not force:
            cached = self._read_cache(key, ttl)
            if cached is not None:
                try:
                    return json.loads(cached.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._purge(key)  # poisoned cache entry: re-fetch once below
        body = self.request(method, url, ttl=ttl, force=True, **kw)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._purge(key)  # never serve the bad body back from cache
            raise SourceBlocked(f"{method} {url} returned a non-JSON body") from exc

    def get_json(self, url: str, *, ttl: float, **kw) -> dict:
        return self._get_json("GET", url, ttl=ttl, **kw)

    def post_json(self, url: str, body: dict, *, ttl: float, **kw) -> dict:
        kw.setdefault("headers", {})
        kw["headers"].setdefault("Content-Type", "application/json")
        return self._get_json("POST", url, ttl=ttl, json_body=body, **kw)

    def set_interval(self, host: str, seconds: float) -> None:
        self.limiter.intervals[host] = seconds
