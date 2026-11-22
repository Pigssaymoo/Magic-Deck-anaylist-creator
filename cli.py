"""Command-line interface.

Usage:
    python -m deckbuilder "Atraxa, Praetors' Voice"
    python -m deckbuilder "Thrasios, Triton Hero" "Tymna the Weaver" --budget 150
    python -m deckbuilder Atraxa --weights '{"edhrec": 0.5, "edhtools": 0.3, "recommander": 0.2}'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, load_config, override_source_weights
from .httpclient import Http
from .pipeline import BuildResult, build


def _split_themes(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _print_summary(result: BuildResult) -> None:
    deck = result.deck
    sel = result.selection
    if sel is not None and sel.winner is not None:
        print(
            f"commander: {sel.winner_name} (auto-selected from {sel.slug!r}; "
            f"score {sel.winner.prelim_score:.4f})"
        )
    print()
    print(f"deck: {', '.join(c.name for c in result.commanders)}")
    if deck.companion:
        print(f"companion: {deck.companion} (outside the 100-card limit)")
    quotas = deck.meta.get("quotas") or {}
    if quotas:
        parts = [f"{k} {v[1]}/{v[0]}" for k, v in quotas.items()]
        print(f"quotas: {' '.join(parts)}")
    est = sum(1 for p in result.prices.values() if p.estimated)
    unpriced = [a.name for a in deck.all_cards if a.name not in result.prices]
    print(
        f"pricing: {len(result.prices)} cards priced, {est} FX-estimated, "
        f"{len(unpriced)} unpriced"
    )
    b = result.budget
    if b is not None:
        verdict = "within budget" if b.within_budget else f"OVER by NZ${b.gap_nzd:,.2f}"
        print(
            f"total: NZ${b.total_nzd:,.2f} "
            f"(cards NZ${b.cards_total_nzd:,.2f} + shipping NZ${b.shipping_nzd:,.2f}) -> {verdict}"
        )
    if result.stock_subs:
        print(f"substitutions: {len(result.stock_subs)} logged")
    if b is not None and b.swaps:
        print(f"budget swaps: {len(b.swaps)} logged")
    print("outputs:")
    for p in result.outputs:
        print(f"  {p}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deckbuilder",
        description=(
            "Build a structurally sound 100-card Commander deck from aggregated "
            "EDHREC / Recommander / edh.tools recommendations, price-matched in "
            "NZD against MTG Singles NZ live stock."
        ),
    )
    parser.add_argument(
        "commanders",
        nargs="*",
        metavar="COMMANDER",
        help="0-2 commander names (fuzzy match; partner = 2 commanders). "
             "Omit and use --theme/--tags to let deckbuilder pick the commander.",
    )
    parser.add_argument("--theme", action="append", default=None, metavar="NAME",
                        help="EDHREC tag/theme slug (repeatable or comma-separated), "
                             "e.g. --theme voltron. Constrain a named commander, or "
                             "without one, select the commander from that tag.")
    parser.add_argument("--tags", default=None, metavar="QUERY",
                        help='Scryfall tag search, e.g. --tags "otag:draw"; selects the '
                             "commander from commander-legal creatures in the results")
    parser.add_argument("--color-identity", default=None, metavar="COLORS",
                        help="required color identity (subset of WUBRG), e.g. WUB; penalizes "
                             "commanders missing a required color")
    parser.add_argument("--candidates", type=int, default=None, metavar="N",
                        help="how many commander candidates to score+probe in theme mode "
                             "(default 5)")
    parser.add_argument("--budget", type=float, default=None, metavar="NZD",
                        help="max total in NZD including the $7.50 flat shipping")
    parser.add_argument("--condition", default=None,
                        help="card condition to require (default: any)")
    parser.add_argument("--on-budget-fail", default="degrade",
                        choices=("degrade", "fail", "emit"), metavar="MODE",
                        help="what to do when the closest deck still exceeds --budget "
                             "after the degradation ladder: 'degrade' (default) fail "
                             "the build with the ladder-exhausted error; 'fail' the "
                             "legacy hard-gate message; 'emit' accept the over-budget "
                             "deck (within_budget=false, loud banner)")
    parser.add_argument("--marketplaces", default=None, metavar="LIST",
                        help="accepted for interface completeness; MTG Singles NZ is the only live price source")
    parser.add_argument("--themes", default=None, metavar="LIST",
                        help="comma-separated theme package names (exactly 4 to override the defaults)")
    parser.add_argument("--weights", default=None, metavar="JSON",
                        help='source weight override, e.g. \'{"edhrec": 0.5, "edhtools": 0.3, "recommander": 0.2}\'')
    parser.add_argument("--owned", default=None, metavar="FILE",
                        help="text file of card names the user already owns (one per line; "
                             "'#' comments and blank lines skipped); names that match the "
                             "card database cost NZ$0.00 and are never probed; unmatched "
                             "names are warned and ignored")
    parser.add_argument("--free-basics", action="store_true",
                        help="basic lands cost NZ$0.00 (the user owns them); the canonical "
                             "basics are derived from the card database")
    parser.add_argument("--min-quality", type=float, default=None, metavar="F",
                        help="minimum consensus score for degradation-ladder replacements "
                             "(R2 swap pass, R3); unranked filler cards are exempt; "
                             "0 disables the floor (default from config: 0.10)")
    parser.add_argument("--companion", default=None, metavar="NAME",
                        help="explicit companion card (recorded outside the 100-card limit)")
    parser.add_argument("--output-dir", default="output", metavar="DIR",
                        help="where to write decklist.txt/json, substitutions.md, report.md (default: output)")
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="path to config.yaml (default: auto-discover)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if len(args.commanders) > 2:
        print("error: at most 2 commanders (partner decks are exactly 2)", file=sys.stderr)
        return 2

    theme_slugs: list[str] = []
    if args.theme:
        for chunk in args.theme:
            theme_slugs.extend(_split_themes(chunk))
    if theme_slugs and args.tags:
        print("error: use either --theme or --tags, not both", file=sys.stderr)
        return 2
    if not args.commanders and not theme_slugs and not args.tags:
        print(
            "error: no commanders given; pass commander name(s) or --theme/--tags "
            "so deckbuilder can select the commander",
            file=sys.stderr,
        )
        return 2
    color_identity: set[str] | None = None
    if args.color_identity:
        color_identity = set(args.color_identity.upper())
        if not color_identity <= set("WUBRG"):
            print(
                f"error: --color-identity must be a non-empty subset of WUBRG "
                f"(got {args.color_identity!r})",
                file=sys.stderr,
            )
            return 2
    if args.candidates is not None and args.candidates < 1:
        print("error: --candidates must be >= 1", file=sys.stderr)
        return 2
    if args.min_quality is not None and args.min_quality < 0:
        print("error: --min-quality must be >= 0 (0 disables the floor)", file=sys.stderr)
        return 2

    owned_names: list[str] | None = None
    if args.owned:
        try:
            text = Path(args.owned).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: could not read --owned file {args.owned!r}: {exc}", file=sys.stderr)
            return 2
        owned_names = [
            ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    try:
        cfg: Config = load_config(args.config)
        cfg = override_source_weights(cfg, args.weights)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.min_quality is not None:
        cfg.budget.min_quality = args.min_quality

    if args.marketplaces and "mtgsingles" not in args.marketplaces.lower():
        print(
            f"note: --marketplaces={args.marketplaces!r} accepted, but MTG Singles NZ "
            "is the only live price source; it will be used.",
            file=sys.stderr,
        )

    http = Http(cfg)
    themes = _split_themes(args.themes) if args.themes else None
    if themes and len(themes) != 4:
        print(
            f"note: {len(themes)} theme names given; exactly 4 are required to override "
            "the defaults, falling back.",
            file=sys.stderr,
        )

    try:
        result = build(
            cfg,
            http,
            args.commanders,
            budget_nzd=args.budget,
            condition=args.condition,
            themes=themes,
            companion_name=args.companion,
            output_dir=args.output_dir,
            verbose=not args.quiet,
            theme_slugs=theme_slugs or None,
            tag_query=args.tags,
            color_identity=color_identity,
            n_candidates=args.candidates,
            on_budget_fail=args.on_budget_fail,
            owned_names=owned_names,
            free_basics=args.free_basics,
        )
    except SystemExit as exc:
        msg = exc.code if isinstance(exc.code, str) else f"commander resolution failed (exit {exc.code})"
        print(f"error: {msg}", file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
