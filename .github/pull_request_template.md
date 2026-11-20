## What changed

## How I checked it
- [ ] `make check` passes locally (lint, security, tests)
- [ ] Opened the Vercel preview link and tried the change
- [ ] New schema? Added an Alembic migration (and to `app/schema_live.py` if it must apply on start)
- [ ] Changed the agent, prompts or model? The "Agent evals" check stays green
