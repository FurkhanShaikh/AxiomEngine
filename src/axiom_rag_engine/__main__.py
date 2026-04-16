"""Axiom Engine command-line interface.

Installed as the `axiom-rag-engine` console script via `[project.scripts]` in
pyproject.toml. Three subcommands:

    axiom-rag-engine serve          Run the FastAPI HTTP server.
    axiom-rag-engine probe "..."    Send a test query to a running server.
    axiom-rag-engine check-config   Print the resolved Settings (secrets redacted).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "axiom_rag_engine.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from axiom_rag_engine.cli.probe import run_probe

    return run_probe(
        query=args.query,
        server_url=args.url,
        model=args.model,
        debug=args.debug,
    )


def _cmd_check_config(args: argparse.Namespace) -> int:
    from axiom_rag_engine.config.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    data = settings.redacted_dict()

    if args.format == "json":
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        return 0

    width = max(len(k) for k in data)
    sys.stdout.write("Axiom Engine — resolved configuration (secrets redacted)\n")
    sys.stdout.write("=" * 60 + "\n")
    for key, value in sorted(data.items()):
        sys.stdout.write(f"  {key:<{width}} = {value!r}\n")
    sys.stdout.write(
        "\nNote: TAVILY_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY "
        "are read directly by vendor SDKs and not shown here.\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom-rag-engine",
        description="Citation-verified RAG service.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the Axiom Engine HTTP server.")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only).")
    serve.set_defaults(func=_cmd_serve)

    probe = sub.add_parser("probe", help="Send a test query to a running server.")
    probe.add_argument("query", help="The question to ask.")
    probe.add_argument("--url", default="http://localhost:8000", help="Server URL.")
    probe.add_argument(
        "--model",
        default="ollama/gemma4:e4b",
        help="LiteLLM model ID for synthesizer + verifier.",
    )
    probe.add_argument("--debug", action="store_true", help="Include audit trail in output.")
    probe.set_defaults(func=_cmd_probe)

    check = sub.add_parser(
        "check-config",
        help="Print the resolved runtime configuration (secrets redacted).",
    )
    check.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    check.set_defaults(func=_cmd_check_config)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
