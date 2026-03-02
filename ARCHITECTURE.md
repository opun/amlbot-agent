# ARCHITECTURE

High-level architecture references:

1. `docs/architecture/overview.md`
2. `docs/db-schema.md`
3. `docs/diagrams.md`

Control Plane (NestJS) handles decisions, approvals, and side effects.
Data Plane (Supabase read APIs) serves read-only reporting via VIEWs + RLS.
