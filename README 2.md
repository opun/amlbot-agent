# MVP Monorepo Blueprint — Next.js + NestJS + Auth0 + Supabase

This archive is a starter blueprint (documentation + repository skeleton) for an MVP where:
- **NestJS** is the **Control Plane** (money, policies, approvals, integrations, audit).
- **Supabase Data API** is the **Data Plane** (strictly **read-only** access to dashboards/reports via **RLS**).
- **Auth0** — user authentication (and later M2M, if needed).
- **Supabase Postgres** — main DB (ORM: Prisma).

## What's inside
- `docs/architecture/overview.md` — unified architecture document
- `docs/mvp-checklist.md` — MVP checklist
- `docs/diagrams.md` — dataflow and userflow diagrams (Mermaid)
- `docs/db-schema.md` — table descriptions and access + RLS
- `Agents.md` and separate AGENTS files for `web/api/supabase`
- `scripts/init.sh` — basic bootstrap (prereqs check + installation)
- `apps/` and `packages/` — monorepo skeleton
- `infra/supabase/migrations/` — SQL for RLS + views (executed via Supabase CLI)

## Quick start
1) Install prereqs: Node.js LTS, pnpm, Docker, Supabase CLI.
2) Run:
   - `bash scripts/init.sh`
3) Fill in `.env.example` → `.env` for `apps/web` and `apps/api`.
4) Connect Supabase project (if using Supabase cloud):
   - `supabase login`
   - `supabase link --project-ref <ref>`
5) Apply Supabase SQL migrations:
   - `cd infra/supabase && supabase db push`

Note: this is a blueprint, not production-ready code. It intentionally keeps code minimal but establishes security rules.

## Unit tests & 100% coverage
- `apps/api`: `pnpm -C apps/api test:cov` (jest + 100% thresholds)
- `apps/web`: `pnpm -C apps/web test:cov` (jest + 100% thresholds)

Note: 100% coverage is a goal and discipline. In real code use:
- small pure functions,
- dependency injection + interfaces (mocks),
- contract tests separate from unit tests, if needed.
