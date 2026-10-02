# Feature map

Drive each with [../SKILL.md](../SKILL.md). Status is the last time it was actually driven.

| Feature | File | Status |
| --- | --- | --- |
| Diagnostics read API | [diagnostics-api.md](diagnostics-api.md) | driven 2026-10-02 |
| Chat turn (router to child agent) | [chat.md](chat.md) | not driven (writes to shared Mongo) |
| Session history and clarifications | [session-history.md](session-history.md) | not driven |
| Diagnostic UI (Lab, Agents, Conversations, Runs) | [diagnostic-ui.md](diagnostic-ui.md) | not driven (frontend not buildable on this network) |

Not implemented by design: auth, streaming, checkpoint resume, summaries (see the root README.md).
