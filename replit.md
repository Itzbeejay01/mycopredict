# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## TB Treatment Outcome Predictor (`tb_app/`)

A standalone Python Flask app for predicting tuberculosis treatment outcomes
(Favorable vs. Unfavorable) using a hard-voting ensemble of Logistic
Regression and Random Forest (SVM optional). Lives outside the pnpm
workspace.

- Run via the `Start application` workflow (`python tb_app/app.py`, port 5000)
- Train models: `python tb_app/train_models.py`
- Models saved as `tb_app/models/{logistic_model,random_forest_model,feature_columns}.pkl`
