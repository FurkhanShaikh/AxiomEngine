"""Send a test query to a running Axiom Engine server.

Shared between ``tasks.py probe`` and ``axiom-rag-engine probe``.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime


def _echo(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


def run_probe(
    query: str,
    server_url: str = "http://localhost:8000",
    model: str = "ollama/gemma4:e4b",
    debug: bool = False,
) -> int:
    """Send *query* to the server and pretty-print the response. Returns 0 on success."""
    payload = {
        "request_id": f"probe-{datetime.now().strftime('%H%M%S')}",
        "user_query": query,
        "models": {"synthesizer": model, "verifier": model},
        "include_debug": debug,
    }

    _echo(f"\n  Query : {query}")
    _echo(f"  Model : {model}")
    _echo(f"  Debug : {debug}")
    _echo(f"  Server: {server_url}\n")

    data = json.dumps(payload).encode()
    req = urllib.request.Request(  # noqa: S310
        f"{server_url}/v1/synthesize",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=700) as resp:  # noqa: S310
            result = json.loads(resp.read())
    except urllib.error.URLError as exc:
        _echo(f"  Could not reach server: {exc}")
        _echo("  Is it running? Try: axiom-rag-engine serve")
        return 1

    # ── Summary ──────────────────────────────────────────────────────────
    status = result.get("status", "?")
    score = result.get("confidence_summary", {}).get("overall_score", 0)
    tiers = result.get("confidence_summary", {}).get("tier_breakdown", {})
    sentences = result.get("final_response", [])

    _echo(f"  Status : {status}")
    _echo(f"  Score  : {score:.2f}")
    _echo(f"  Tiers  : { {k: v for k, v in tiers.items() if v > 0} or 'none' }")
    if result.get("error_message"):
        _echo(f"  Error  : {result['error_message']}")
    _echo()

    for i, s in enumerate(sentences, 1):
        vr = s.get("verification", {})
        _echo(f"  [{i}] Tier {vr.get('tier', '?')} ({vr.get('tier_label', '?')}) — {s['text']}")
        for c in s.get("citations", []):
            _echo(f'       cite: "{c["exact_source_quote"][:80]}"')
            _echo(f"       from: {c['chunk_id']}")

    # ── Debug info ────────────────────────────────────────────────────────
    if debug and result.get("debug"):
        dbg = result["debug"]
        stats = dbg.get("pipeline_stats", {})
        _echo(f"\n  Pipeline stats: {stats}")
        _echo(f"\n  Audit trail ({len(dbg.get('audit_trail', []))} events):")
        for event in dbg.get("audit_trail", []):
            payload_str = json.dumps(event.get("payload", {}))
            if len(payload_str) > 120:
                payload_str = payload_str[:120] + "…"
            _echo(f"    [{event['node']}] {event['event_type']}: {payload_str}")

    return 0
