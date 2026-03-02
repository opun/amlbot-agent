# CLAUDE.md

This file is a thin adapter for Claude Code.

## Source of Truth

- Read `AGENTS.md` first.
- Read `skills.md` second.
- Then load only the relevant docs from `docs/` for the current task.

## Required Execution Loop

Follow the loop defined in `AGENTS.md`:

1. Context load
2. Plan
3. Implement
4. Self-review
5. Validate
6. Document
7. Report

## Policy

Do not duplicate policy in this file. If a rule changes, update `AGENTS.md` or `skills.md`.
