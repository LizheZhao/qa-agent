# Adding a tool

Add a module beneath `packages/tools/src/orchestration_tools/<family>/`, not inside an agent. The
`orchestration_tools` directory is the installed import package, while `<family>` groups related tools.
The `orchestration-tools` distribution contains the worker's internal tool modules; do not create a
distribution per tool. Choose a globally stable name and LLM-useful description, define a Pydantic
input schema, and implement an async operation with explicit runtime dependencies and normalized
`ToolResult`/`ToolError` behavior. Convert it to a LangChain `StructuredTool` so standard model tool
calls and LangGraph `ToolNode` can execute it.

Tools call API-service clients. They never connect directly to MongoDB or another domain database.
Test with fake clients: schema validation, success, normalized errors, and `ToolNode` integration. Tool
registry implementation remains internal to the application. The tools distribution is pinned by
`uv.lock` and is not selected independently in the deployment manifest.
