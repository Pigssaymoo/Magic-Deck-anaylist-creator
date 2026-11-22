"""EDHREC source: commander pages, tag/theme pages (theme mode).

Endpoints (live-verified):
  commander  https://json.edhrec.com/pages/commanders/{slug}.json
             top-level ``tag_counts`` (DESC); ``container.json_dict.card`` has
             name/slug/num_decks/names. NO ``related_info`` on commander pages.
  tag        https://json.edhrec.com/pages/tags/{slug}.json
  theme      https://json.edhrec.com/pages/themes/{slug}.json
Tag/theme pages carry ONLY ``container.json_dict.cardlists``; each cardlist is
``{header, tag, cardviews}``. Cardviews carry name/slug/num_decks/
potential_decks; card-pool views add synergy/lift; partner commander views add
``is_partner``, ``names: [A, B]`` and ``cards: [{name, url}]``.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .httpclient import Http, SourceBlocked

PAGE_URL = "https://json.edhrec.com/pages/commanders/{slug}.json"
TAG_PAGE_URL = "https://json.edhrec.com/pages/tags/{slug}.json"
THEME_PAGE_URL = "https://json.edhrec.com/pages/themes/{slug}.json"
TAG_INDEX_URL = "https://edhrec.com/tags"
THEME_INDEX_URL = "https://edhrec.com/themes"

# Cardlist tags that hold commander candidates on tag/theme pages.
COMMANDER_LIST_TAGS = ("topcommanders", "newcommanders")

# Cardlist tags that map to the four theme packages (in package order).
TAG_PACKAGE_MAP = (
    ("creatures", 0),
    ("instants", 1),
    ("sorceries", 1),
    ("enchantments", 2),
    ("planeswalkers", 2),
    ("utilityartifacts", 3),
    ("utilitylands", 3),
    ("manaartifacts", 3),
)


def slugify(name: str) -> str:
    """'Atraxa, Praetors' Voice' -> 'atraxa-praetors-voice'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass
class EdhRecSignal:
    name: str
    inclusion: float = 0.0          # num_decks / potential_decks
    synergy: float = 0.0
    lift: float = 0.0
    trend_zscore: float = 0.0
    norm_inclusion: float = 0.0     # percentile within the response
    norm_synergy: float = 0.0
    norm_lift: float = 0.0
    rank: int = 0                   # rank by inclusion (1 = best)
    n: int = 0                      # size of the response
    lists: list[str] = field(default_factory=list)
    package_hint: int | None = None


@dataclass
class EdhrecData:
    commander_name: str
    signals: dict[str, EdhRecSignal] = field(default_factory=dict)  # keyed by raw name
    mana_curve: dict[int, int] = field(default_factory=dict)
    list_tags: list[str] = field(default_factory=list)


def _percentile(sorted_desc: list, rank: int, n: int) -> float:
    """rank 1 (best) -> ~1.0, worst -> ~1/n. n == 1 -> 1.0."""
    if n <= 0:
        return 0.0
    return (n - rank + 1) / n


def _root(data: dict[str, Any]) -> dict[str, Any]:
    """Accept both payload generations: current top-level shape and the old
    ``{"data": {...}}``-wrapped shape (kept for cached responses + tests)."""
    inner = data.get("data")
    return inner if isinstance(inner, dict) else data


def parse_page(data: dict[str, Any], commander_name: str) -> EdhrecData:
    """Parse a raw EDHREC commander-page payload (pure, unit-testable)."""
    root = _root(data)
    jd = (root.get("container") or {}).get("json_dict", {}) or {}
    cmd = jd.get("card") or {}
    if commander_name and cmd.get("name"):
        commander_name = cmd["name"]

    # ---- collect raw per-card records, merging across lists
    raw: dict[str, dict[str, Any]] = {}
    list_tags: list[str] = []
    package_hint: dict[str, int] = {}
    for cl in jd.get("cardlists", []) or []:
        tag = (cl.get("tag") or "").lower()
        if tag:
            list_tags.append(tag)
        for cv in cl.get("cardviews", []) or []:
            name = (cv.get("name") or "").strip()
            if not name:
                continue
            inc = 0.0
            try:
                potential = float(cv.get("potential_decks") or 0)
                if potential > 0:
                    inc = float(cv.get("num_decks") or 0) / potential
            except (TypeError, ValueError):
                inc = 0.0
            rec = raw.get(name)
            if rec is None:
                rec = {
                    "inclusion": inc,
                    "synergy": float(cv.get("synergy") or 0),
                    "lift": float(cv.get("lift") or 0),
                    "trend": float(cv.get("trend_zscore") or 0),
                    "lists": [],
                }
                raw[name] = rec
            else:
                rec["inclusion"] = max(rec["inclusion"], inc)
                rec["synergy"] = max(rec["synergy"], float(cv.get("synergy") or 0))
                rec["lift"] = max(rec["lift"], float(cv.get("lift") or 0))
                rec["trend"] = max(rec["trend"], float(cv.get("trend_zscore") or 0))
            if tag and tag not in rec["lists"]:
                rec["lists"].append(tag)
            for known, pkg in TAG_PACKAGE_MAP:
                if tag == known and name not in package_hint:
                    package_hint[name] = pkg

    # ---- rank-normalize each signal within this response
    n = len(raw)
    by_incl = sorted(raw.items(), key=lambda kv: (-kv[1]["inclusion"], kv[0].casefold()))
    by_syn = sorted(raw.items(), key=lambda kv: (-kv[1]["synergy"], kv[0].casefold()))
    by_lift = sorted(raw.items(), key=lambda kv: (-kv[1]["lift"], kv[0].casefold()))
    rank_incl = {name: i + 1 for i, (name, _) in enumerate(by_incl)}
    rank_syn = {name: i + 1 for i, (name, _) in enumerate(by_syn)}
    rank_lift = {name: i + 1 for i, (name, _) in enumerate(by_lift)}

    signals: dict[str, EdhRecSignal] = {}
    for name, rec in raw.items():
        signals[name] = EdhRecSignal(
            name=name,
            inclusion=rec["inclusion"],
            synergy=rec["synergy"],
            lift=rec["lift"],
            trend_zscore=rec["trend"],
            norm_inclusion=_percentile(by_incl, rank_incl[name], n),
            norm_synergy=_percentile(by_syn, rank_syn[name], n),
            norm_lift=_percentile(by_lift, rank_lift[name], n),
            rank=rank_incl[name],
            n=n,
            lists=list(rec["lists"]),
            package_hint=package_hint.get(name),
        )

    panels = (root.get("panels") or {}) or {}
    curve: dict[int, int] = {}
    for k, v in (panels.get("mana_curve") or {}).items():
        try:
            curve[int(float(k))] = int(v)
        except (TypeError, ValueError):
            continue

    return EdhrecData(commander_name=commander_name, signals=signals, mana_curve=curve, list_tags=list_tags)


def _fetch_page(http: Http, cfg: Config, slug: str) -> dict[str, Any]:
    url = PAGE_URL.format(slug=slug)
    return http.get_json(url, ttl=cfg.http.rec_cache_days * 24 * 3600)


def merge_pages(a: EdhrecData, b: EdhrecData) -> EdhrecData:
    """Merge two solo pages (partner fallback): keep the best signal per card."""
    out = EdhrecData(
        commander_name=f"{a.commander_name} + {b.commander_name}",
        mana_curve=dict(a.mana_curve or b.mana_curve),
        list_tags=list(dict.fromkeys(a.list_tags + b.list_tags)),
    )
    names = set(a.signals) | set(b.signals)
    for name in names:
        sa, sb = a.signals.get(name), b.signals.get(name)
        if sa is None:
            out.signals[name] = sb
            continue
        if sb is None:
            out.signals[name] = sa
            continue
        out.signals[name] = EdhRecSignal(
            name=name,
            inclusion=max(sa.inclusion, sb.inclusion),
            synergy=max(sa.synergy, sb.synergy),
            lift=max(sa.lift, sb.lift),
            trend_zscore=max(sa.trend_zscore, sb.trend_zscore),
            norm_inclusion=max(sa.norm_inclusion, sb.norm_inclusion),
            norm_synergy=max(sa.norm_synergy, sb.norm_synergy),
            norm_lift=max(sa.norm_lift, sb.norm_lift),
            rank=min(sa.rank, sb.rank),
            n=max(sa.n, sb.n),
            lists=list(dict.fromkeys(sa.lists + sb.lists)),
            package_hint=sa.package_hint if sa.package_hint is not None else sb.package_hint,
        )
    return out


def fetch(http: Http, cfg: Config, commander_names: list[str]) -> EdhrecData | None:
    """Fetch EDHREC data for 1 commander or a partner pair (solo merge fallback)."""
    if len(commander_names) == 2:
        slug = f"{slugify(commander_names[0])}-{slugify(commander_names[1])}"
        try:
            return parse_page(_fetch_page(http, cfg, slug), commander_names[0])
        except SourceBlocked:
            pass  # fall through to solo merge
    pages: list[EdhrecData] = []
    for name in commander_names:
        try:
            pages.append(parse_page(_fetch_page(http, cfg, slugify(name)), name))
        except SourceBlocked:
            continue
    if not pages:
        raise SourceBlocked("EDHREC: no commander page could be fetched")
    data = pages[0]
    for extra in pages[1:]:
        data = merge_pages(data, extra)
    return data


# ==================================================================== theme
@dataclass
class ThemeCommander:
    """One commander entry from a tag/theme page's Top/New Commanders list."""

    name: str  # canonical display name (partners: "A, Title // B, Title")
    slug: str
    num_decks: int = 0
    potential_decks: int = 0
    is_partner: bool = False
    partner_names: tuple[str, ...] = ()  # individual partner names

    @property
    def inclusion(self) -> float:
        if self.potential_decks <= 0:
            return 0.0
        return self.num_decks / self.potential_decks


@dataclass
class ThemeCard:
    """One card from a tag/theme page's card-pool lists (Top/High Lift/...)."""

    name: str
    synergy: float = 0.0
    lift: float = 0.0
    num_decks: int = 0
    list_tags: list[str] = field(default_factory=list)


@dataclass
class ThemePage:
    """Parsed EDHREC tag or theme page: commander candidates + theme card pool."""

    slug: str
    kind: str  # "tag" | "theme"
    url: str
    name: str = ""  # human-readable tag/theme name (page "header")
    commanders: list[ThemeCommander] = field(default_factory=list)
    cards: list[ThemeCard] = field(default_factory=list)

    @property
    def card_names(self) -> set[str]:
        return {c.name for c in self.cards}


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_theme_page(data: dict[str, Any], *, slug: str, kind: str, url: str) -> ThemePage:
    """Parse a raw EDHREC tag/theme page payload (pure, unit-testable).

    Commander lists (tag ``topcommanders``/``newcommanders``) become
    candidates; every other cardlist contributes to the theme card pool.
    """
    root = _root(data)
    jd = (root.get("container") or {}).get("json_dict", {}) or {}
    commanders: list[ThemeCommander] = []
    seen_slugs: set[str] = set()
    cards: dict[str, ThemeCard] = {}
    for cl in jd.get("cardlists") or []:
        tag = (cl.get("tag") or "").lower()
        if not tag:
            continue
        for cv in cl.get("cardviews") or []:
            name = str(cv.get("name") or "").strip()
            if not name:
                continue
            if tag in COMMANDER_LIST_TAGS:
                cslug = str(cv.get("slug") or slugify(name))
                if cslug in seen_slugs:
                    continue  # topcommanders/newcommanders dedupe by slug
                seen_slugs.add(cslug)
                partner_names: tuple[str, ...] = ()
                for n in cv.get("names") or []:
                    n = str(n).strip()
                    if n:
                        partner_names += (n,)
                if not partner_names:
                    for c in cv.get("cards") or []:
                        n = str((c or {}).get("name") or "").strip()
                        if n:
                            partner_names += (n,)
                commanders.append(
                    ThemeCommander(
                        name=name,
                        slug=cslug,
                        num_decks=_int(cv.get("num_decks")),
                        potential_decks=_int(cv.get("potential_decks")),
                        is_partner=bool(cv.get("is_partner")),
                        partner_names=partner_names,
                    )
                )
            else:
                rec = cards.get(name)
                if rec is None:
                    rec = ThemeCard(name=name)
                    cards[name] = rec
                try:
                    rec.synergy = max(rec.synergy, float(cv.get("synergy") or 0))
                except (TypeError, ValueError):
                    pass
                try:
                    rec.lift = max(rec.lift, float(cv.get("lift") or 0))
                except (TypeError, ValueError):
                    pass
                rec.num_decks = max(rec.num_decks, _int(cv.get("num_decks")))
                if tag not in rec.list_tags:
                    rec.list_tags.append(tag)
    name = str(root.get("header") or root.get("name") or slug).strip()
    return ThemePage(
        slug=slug, kind=kind, url=url, name=name, commanders=commanders, cards=list(cards.values())
    )


def _close_slugs(http: Http, slug: str, limit: int = 5) -> list[str]:
    """Best-effort 'did you mean' from the edhrec.com tag/theme index pages."""
    slugs: list[str] = []
    for url in (TAG_INDEX_URL, THEME_INDEX_URL):
        try:
            body = http.request("GET", url)
        except Exception:  # noqa: BLE001 - suggestions are best-effort only
            continue
        text = body.decode("utf-8", "replace")
        for s in re.findall(r'href="/(?:tags|themes)/([a-z0-9\-]+)"', text):
            if s not in slugs:
                slugs.append(s)
    slugs = [s for s in slugs if s != slug]
    return difflib.get_close_matches(slug, slugs, n=limit, cutoff=0.5)


def resolve_theme_page(http: Http, cfg: Config, slug: str) -> ThemePage:
    """Resolve a theme name to an EDHREC tag page (themes fallback).

    S3 "AccessDenied" (HTTP 403) on the JSON API is treated as 404. When the
    slug exists in neither, pull the index pages and exit with a
    "did you mean" list — never silently guess.
    """
    slug = slugify(str(slug).strip())
    if not slug:
        raise SystemExit("error: empty theme name")
    failures: list[str] = []
    for url_tpl, kind in ((TAG_PAGE_URL, "tag"), (THEME_PAGE_URL, "theme")):
        url = url_tpl.format(slug=slug)
        try:
            data = http.get_json(url, ttl=cfg.http.rec_cache_days * 24 * 3600)
        except SourceBlocked as exc:
            failures.append(f"{url}: {exc}")
            continue
        page = parse_theme_page(data, slug=slug, kind=kind, url=url)
        if page.commanders or page.cards:
            return page
        failures.append(f"{url}: empty page")
    suggestions = _close_slugs(http, slug)
    tail = "; did you mean: " + ", ".join(suggestions) if suggestions else " (no close matches found)"
    raise SystemExit(f"error: no EDHREC tag or theme named {slug!r}{tail}")
