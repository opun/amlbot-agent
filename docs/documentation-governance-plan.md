# Documentation Governance Plan (AGENTS + Skills + Persistent Memory)

## Objective

Adopt a predictable markdown documentation system so every Claude/Codex interaction can produce durable, reviewable artifacts.

## Target Structure

```text
AGENTS.md
ARCHITECTURE.md
docs/
├── design-docs/
│   ├── index.md
│   ├── core-beliefs.md
│   └── ...
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── generated/
│   └── db-schema.md
├── product-specs/
│   ├── index.md
│   ├── new-user-onboarding.md
│   └── ...
├── references/
│   ├── design-system-reference-llms.txt
│   ├── nixpacks-llms.txt
│   ├── uv-llms.txt
│   └── ...
├── DESIGN.md
├── FRONTEND.md
├── PLANS.md
├── PRODUCT_SENSE.md
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md
```

## Rollout Steps

1. Canonicalize root governance file as `AGENTS.md` and keep one source of truth.
2. Create indexes for `design-docs` and `product-specs`.
3. Move active implementation plans into `docs/exec-plans/active`.
4. On completion, archive plans into `docs/exec-plans/completed` with outcome summary.
5. Maintain generated artifacts in `docs/generated` from scripts only.

## What to add in `AGENTS.md`

Add a **Documentation Persistence Contract** section:

1. For feature requests, agent must create/update an exec plan before code changes.
2. For architecture decisions, agent must append/update a design doc or ADR entry.
3. For product behavior changes, agent must update product spec index + relevant spec.
4. For each session, agent logs artifacts touched in a short “Session Notes” section.
5. No PR is complete without doc links and verification commands.

Recommended enforceable wording:

- "If implementation changes and no matching doc update exists, task is incomplete."
- "When uncertain where to write context, default to `docs/exec-plans/active/<date>-<topic>.md`."

## What to add in `skills.md`

Add a new skill: `docs-persistence`.

### docs-persistence

**Use when:** any non-trivial coding, planning, architecture, or product clarification task.

**Procedure**

1. Classify request type: feature, bug, architecture, ops, or research.
2. Route output to required doc location:
   - feature/bug -> `docs/exec-plans/active/`
   - architecture -> `docs/design-docs/`
   - product -> `docs/product-specs/`
   - generated schema/report -> `docs/generated/`
3. Add/update front-matter: `status`, `owner`, `last_updated`, `related_prs`.
4. Append a “Decision Log” entry with date, decision, rationale, and validation.
5. If task completes, move/mark plan as completed and summarize residual debt.

**Deliverables**

- Updated doc artifact(s)
- Index links refreshed
- Decision log entry

## Make it enforceable in CI

1. Add `scripts/docs/check-doc-updates.ts`:
   - detect changed code paths
   - require corresponding docs path change
2. Add `scripts/docs/check-frontmatter.ts`:
   - validate required fields
3. Add CI step `pnpm docs:check` as merge gate.
4. Add PR template checkboxes for plan/spec/design updates.

## Suggested Starter Files

1. `AGENTS.md` (canonical root instructions)
2. `ARCHITECTURE.md` (top-level architecture map)
3. `docs/design-docs/index.md`
4. `docs/product-specs/index.md`
5. `docs/exec-plans/tech-debt-tracker.md`
6. `docs/PLANS.md` (navigation page to active/completed plans)
