# AGENTS.md

## Mission

Deliver high-confidence changes with low regression risk by treating this repository as the system of record and using a repeatable execution harness that works across coding agents and models.

## Operating Model (Harness First)

- Humans set intent, scope, and acceptance criteria.
- Agents execute through a strict loop: plan -> implement -> verify -> report.
- Repository-local artifacts are authoritative; off-repo context is non-binding until written into `docs/`.

## Non-Negotiable Rules

1. **Docs first:** Before any non-trivial change, read the relevant doc in the Knowledge Map (or search `docs/`).
2. **Plan first:** Publish scope, assumptions, risks, and verification commands before implementation.
3. **Use existing primitives first:** Reuse shared UI/components/tokens and shared API/domain abstractions.
4. **Narrow changes:** Keep PRs focused on one problem family.
5. **Prove outcomes:** Run required checks and include objective evidence for behavior changes.
6. **Persist knowledge:** Behavior/architecture changes must include doc updates.
7. **Escalate uncertainty:** If requirements conflict, safety is unclear, or data shape is unknown, stop and ask.

## Execution Contract (All Agents)

### Inputs Required

- Task objective and acceptance criteria.
- Impacted domains/files and constraints.
- Verification commands and expected outcomes.

### Required Loop

1. **Context load:** Read `AGENTS.md`, then relevant docs, then `skills.md`.
2. **Plan:** Write a concise plan with assumptions and risks.
3. **Implement:** Make minimal, scoped changes.
4. **Self-review:** Verify diffs against requirements and invariants.
5. **Validate:** Run required commands + targeted manual checks.
6. **Document:** Update canonical docs for behavior/architecture changes.
7. **Report:** Return command outcomes, residual risks, and follow-ups.

### Done Criteria

- Acceptance criteria are met.
- Required verification commands pass.
- Required docs are updated.
- Residual risks/deferred items are explicitly listed.

## Validation Gates

### Frontend / UI

1. Run `ux-audit` from `skills.md`.
2. Apply `style-harmonization` constraints.
3. Run `regression-qa` commands + manual matrix.
4. Complete `docs/testing/ui-ux-release-checklist.md`.
5. Attach before/after screenshots for perceptible visual changes.

### Backend / API

1. Read `docs/architecture/overview.md`.
2. Enforce `docs/SECURITY.md` invariants.
3. If schema changes, update `docs/db-schema.md`.
4. Run `pnpm -F api run lint && pnpm -F api run build`.

## Required PR Deliverables

- Scope and user-facing summary.
- Commands executed and outcomes.
- Manual QA matrix references (when applicable).
- Screenshot evidence (when visuals changed).
- Explicit residual risk / deferred items list.

## Accessibility Baseline

- Keyboard-only flow remains usable.
- Focus-visible styles are preserved or improved.
- Error/helper copy is actionable.
- Destructive operations have explicit warning affordances.

## Repository Knowledge Map

The `docs/` directory is the system of record. Start with the smallest relevant doc and progressively disclose deeper context.

| Area | Path | Description |
|---|---|---|
| Frontend architecture | `docs/FRONTEND.md` | Stack, styling conventions, target architecture |
| Design system | `docs/design-docs/design-system.md` | Tokens, typography, status colors, brand colors |
| Refactoring architecture | `docs/design-docs/web-refactoring-architecture.md` | Target layering, data flow, permission model |
| Refactoring plan | `docs/exec-plans/completed/web-refactoring-plan.md` | Phased frontend execution plan with acceptance criteria |
| API enhancement plan | `docs/exec-plans/completed/api-enhancement-plan.md` | API hardening, OpenAPI, pagination, session permissions, testing |
| Quality grades | `docs/QUALITY_SCORE.md` | Per-domain quality scores and gap analysis |
| Design principles | `docs/DESIGN.md` | Design principles and UX constraints |
| Product specs | `docs/product-specs/index.md` | Product behavior specifications |
| Design docs | `docs/design-docs/index.md` | Architecture decision records |
| Active plans | `docs/exec-plans/active/` | In-progress execution plans |
| Completed plans | `docs/exec-plans/completed/` | Finished execution plans with outcomes |
| Tech debt | `docs/exec-plans/tech-debt-tracker.md` | Tracked technical debt items |
| Security | `docs/SECURITY.md` | Security model and constraints |
| DB schema | `docs/db-schema.md` | Database schema reference |
| RBAC model | `docs/rbac-key-quorum-model.md` | Role-based access control and key quorum design |
| Architecture overview | `docs/architecture/overview.md` | MVP architecture, control plane vs data plane, stack constraints |
| UX audit template | `docs/testing/ui-ux-audit-template.md` | Checklist template for UI/UX audits |

## Documentation Persistence Contract

1. For non-trivial requests, create or update a markdown artifact before finishing.
2. Route artifacts to canonical paths:
   - execution work -> `docs/exec-plans/active/`
   - architecture reasoning -> `docs/design-docs/`
   - product behavior -> `docs/product-specs/`
   - generated snapshots -> `docs/generated/`
3. Include verification commands and outcomes in updated docs.
4. When execution work is complete, move plans to `docs/exec-plans/completed/` and capture residual risk.
5. If implementation changes without matching docs updates, task is incomplete.

## Adapter Files (Tool-Specific Entry Points)

- Cursor rules must delegate to this file plus `skills.md` and relevant docs.
- `CLAUDE.md` must be a thin adapter that points to this file.
- OpenCode config/instructions must reference this file.
- Keep policy in one place: this file and `docs/`.
