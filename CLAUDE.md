# CLAUDE.md

Claude Code adapter for the StableRail monorepo. Cross-agent policy (execution loop, artifact lifecycle, failure-pattern rules, verification commands) lives in `AGENTS.md`. This file adds only Claude Code-specific capabilities.

## Context Loading (Do This First)

1. Read `AGENTS.md` — execution contract, artifact lifecycle, task-type routing, failure-pattern rules.
2. Read `skills.md` — repeatable workflows with verification commands.
3. Load **only** the docs prescribed by the task-type routing table in `AGENTS.md`.

**Context is scarce.** Do not load every doc. Load the minimum set needed, then progressively disclose if you hit unknowns.

## Execution Loop

Follow the Required Loop in `AGENTS.md` with these Claude Code-specific behaviors:

1. **Context load** — Use the routing table in AGENTS.md. Use subagents (`Agent` tool with `subagent_type=Explore`) for broad codebase exploration; use `Grep`/`Glob` for directed searches. Check `docs/exec-plans/active/` for in-progress work.
2. **Plan** — Create the exec-plan artifact per the Session Artifact Lifecycle in AGENTS.md. Use `EnterPlanMode` for multi-step tasks that need user alignment.
3. **Implement** — Update the exec-plan's `## Changes` section as you go.
4. **Self-review** — Diff changes against acceptance criteria in the exec-plan.
5. **Validate** — Run verification commands from AGENTS.md. Record output in the exec-plan.
6. **Document** — Move completed exec-plan to `docs/exec-plans/completed/`.
7. **Report** — Everything should already be in the exec-plan artifact.

## Claude Code Capabilities to Use

- **Subagents** — Parallelize independent research. Use `Explore` agents for broad searches, `Plan` agents for architecture decisions.
- **Hooks** — Pre-commit hooks enforce linting and type-checking. If a hook fails, fix the issue — never bypass with `--no-verify`.
- **Memory** — Use the persistent memory system for cross-session context about user preferences, project state, and past feedback. When a task spans sessions, save a memory note pointing to the active exec-plan.
- **MCP tools** — Supabase for DB operations, Notion for project docs, Figma for design implementation, Postman for API testing.

## Cross-Session Continuity (Claude Code-specific)

If a task spans multiple sessions:
- The exec-plan in `docs/exec-plans/active/` is the handoff document.
- Save a memory note (via the Memory system) pointing to the active plan so the next session can pick up.
- On resume, read the active plan first before loading other context.

## Policy

All cross-agent policy lives in `AGENTS.md`: execution contract, session artifact lifecycle, task-type routing, verification commands, failure-pattern rules, anti-patterns. Do not duplicate here. If a rule changes, update `AGENTS.md`.
