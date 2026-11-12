#!/usr/bin/env python3
"""Write app/static/voices.json from the Python voice bank.

The static build has no Python process, so the browser reads this file instead
of calling /api/voices. The shape matches the endpoint exactly, and
tests/test_api.py fails if the two drift. Run this after editing
app/data/voices.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import voices  # noqa: E402  (needs the path above)
from app.voices import SCHEMA_VERSION  # noqa: E402

OUT = ROOT / "app" / "static" / "voices.json"


def main() -> int:
    bank = voices.bank()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sample_rate": bank.sample_rate,
        "categories": bank.categories(),
        "defaults": bank.defaults,
        "bounds": {k: list(v) for k, v in voices.PARAM_BOUNDS.items()},
        "voices": bank.list_dicts(),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(payload['voices'])} voices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
