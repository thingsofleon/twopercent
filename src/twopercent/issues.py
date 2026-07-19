"""Hardened GitHub issue filing via `gh`, shared by routine score mode
(auto-degradation) and the research runner (promotion-candidate).

Security posture, unchanged from its origin in routine.py:

- subprocess argument LISTS only, never shell=True — titles/labels can never
  be shell-interpreted.
- Body via stdin (`--body-file -`), so arbitrary markdown never touches argv.
- Open-issue dedup by label: at most one open issue per label at a time.
- Label ensured idempotently (`gh label create --force`) before filing.
- Conversation locked at creation: on a public repo, non-collaborators must
  not be able to inject instructions via comments into an issue an agent will
  read (agents treat the machine-generated body as the only trusted input
  regardless).
- Every failure is loud: a warning is logged and a FAILED/LOCK_FAILED result
  returned — filing never raises, and never pretends success.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FILED = "filed"
DUPLICATE = "duplicate"
LOCK_FAILED = "lock_failed"
FAILED = "failed"


@dataclass
class IssueResult:
    outcome: str  # FILED / DUPLICATE / LOCK_FAILED / FAILED
    url: str = ""  # set for FILED and LOCK_FAILED
    existing: str = ""  # "#12, #13" for DUPLICATE
    error: str = ""  # set for LOCK_FAILED and FAILED


def file_issue(label: str, title: str, body: str, color: str, description: str) -> IssueResult:
    """File a labeled, locked GitHub issue unless one with `label` is already open."""
    try:
        listing = subprocess.run(
            ["gh", "issue", "list", "--state", "open", "--label", label, "--json", "number,title"],
            check=True,
            capture_output=True,
            text=True,
        )
        existing = json.loads(listing.stdout or "[]")
        if existing:
            numbers = ", ".join(f"#{item['number']}" for item in existing)
            logger.warning(
                "open %s issue already filed (%s) — not filing a duplicate", label, numbers
            )
            return IssueResult(DUPLICATE, existing=numbers)
        subprocess.run(
            [
                "gh",
                "label",
                "create",
                label,
                "--force",
                "--color",
                color,
                "--description",
                description,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        created = subprocess.run(
            ["gh", "issue", "create", "--title", title, "--label", label, "--body-file", "-"],
            input=body,
            check=True,
            capture_output=True,
            text=True,
        )
        url = created.stdout.strip()
        try:
            number = url.rstrip("/").rsplit("/", 1)[-1]
            if not number.isdigit():
                raise ValueError(f"could not parse issue number from {url!r}")
            subprocess.run(
                ["gh", "issue", "lock", number, "--reason", "off_topic"],
                check=True,
                capture_output=True,
                text=True,
            )
            return IssueResult(FILED, url=url)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
            err = (getattr(exc, "stderr", "") or str(exc)).strip()
            logger.warning("filed %s but could not lock the conversation (%s)", url, err)
            return IssueResult(LOCK_FAILED, url=url, error=err)
    except FileNotFoundError:
        logger.warning("gh CLI not found — NO %s issue was filed", label)
        return IssueResult(FAILED, error="gh CLI not found")
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        err = (getattr(exc, "stderr", "") or str(exc)).strip()
        logger.warning("gh failed (%s) — NO %s issue was filed", err, label)
        return IssueResult(FAILED, error=err)
