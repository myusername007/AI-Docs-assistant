# AI Docs Assistant

Agent that answers questions over internal company documentation:
RAG with access control, tool calling, multi-provider LLMs, deployed and observable.

## Phase 0 — runnable scaffold

```bash
cp .env.example .env
docker compose up --build
```

- Liveness: `GET /health/live`
- Readiness (DB ping): `GET /health/ready`
