"""Which code actually ran — git identity for ledger rows and runner digests.

The scheduled units execute `uv run twopercent ...` in the checkout, which runs
WHATEVER branch is sitting there. That is #114's failure: a feature branch left
checked out ran seven nights of production research, recorded 56 experiments
from unreviewed code, and nothing anywhere — routine digest, research summary,
experiments ledger — could say so afterward. The reviewer found it only by
querying run timestamps against the branch's creation date.

Two consumers, matching the issue's two fixes:
  * `backtest.run_benchmark` records `git_state()` in every experiments row's
    params (sibling of `device`/`feature_set`), so a row is always traceable to
    the code that produced it and "these N rows came from an unmerged commit"
    is one ledger query.
  * The routine and research runners emit a `code` step: OK only on a CLEAN
    `main`, WARN otherwise — including when git state is UNKNOWABLE, because
    "cannot verify" must never render as "verified" (silent success is the
    enemy, and this module exists because a silence cost a week).

Everything degrades to None rather than raising: provenance must never be the
reason a production run dies.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

PRODUCTION_BRANCH = "main"
_GIT_TIMEOUT_S = 5


@dataclass(frozen=True)
class GitState:
    commit: str | None  # full hash — unambiguous in the ledger; describe() shortens
    branch: str | None  # "HEAD" when detached
    dirty: bool | None

    @property
    def is_clean_production(self) -> bool:
        """True only for a verified clean checkout of the production branch.

        Unknown state is NOT production: a run that cannot prove what code it
        executed gets the warning, not the benefit of the doubt.
        """
        return self.branch == PRODUCTION_BRANCH and self.dirty is False

    def describe(self) -> str:
        if self.commit is None:
            return "git state unknowable (no git, or not a checkout)"
        short = self.commit[:9]
        # A None field must render as UNKNOWN, never as its innocent value: a
        # timed-out `git status` is not a clean tree, and printing "clean" next
        # to a WARN about unverifiable code would contradict the warning itself.
        branch = self.branch if self.branch is not None else "branch unknown"
        if self.dirty is None:
            tree = "dirty-state unknown"
        elif self.dirty:
            tree = "dirty tree"
        else:
            tree = "clean"
        return f"{branch} @ {short}, {tree}"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            # Git permits non-UTF8 branch names; strict decoding would raise
            # UnicodeDecodeError (a ValueError) PAST the except below and make
            # provenance the thing that kills a production run — the one
            # failure this module's contract forbids.
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> GitState:
    """The current checkout's identity, from the process working directory.

    The systemd units set WorkingDirectory to the repo, and interactive runs
    start there, so the process CWD is the right anchor — the same resolution
    `store.DEFAULT_DB_PATH` already relies on.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return GitState(None, None, None)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    porcelain = _git("status", "--porcelain")
    dirty = None if porcelain is None else bool(porcelain)
    return GitState(commit, branch, dirty)
