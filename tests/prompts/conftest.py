"""Pytest config shared across tests/prompts/*.

Exposes the ``prompt_case`` parametrized fixture (wired from each
subdirectory's ``test_*.py``) and a ``skip_if_not_runnable`` helper that
honors the live/recorded mode split set up in :mod:`harness`.
"""
from __future__ import annotations

import pytest

from tests.prompts.harness import Case, is_live_mode


def skip_if_case_not_runnable(case: Case) -> None:
    """Skip the test unless the case can actually execute in the current mode.

    * Live mode needs ``OPENAI_API_KEY``.
    * Recorded mode needs the sibling ``.recording.jsonl`` file.

    Either way, we'd rather pytest-skip than pytest-error so CI runs
    without LLM credentials stay green on a freshly-seeded case.
    """
    import os
    if is_live_mode():
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("AGENT_EVAL_LIVE=1 but OPENAI_API_KEY is not set")
        return
    if not case.recording_path.exists():
        pytest.skip(
            f"No recording for {case.name} (expected at {case.recording_path.name}); "
            f"set AGENT_EVAL_LIVE=1 to run live instead"
        )
