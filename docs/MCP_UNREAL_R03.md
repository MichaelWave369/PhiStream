# PhiStream MCP R0.3: Unreal 5.9 loopback discovery pilot

Status: **experimental local protocol adapter; discovery only**.

This is a bridge between PhiStream's R0.1 stdio gateway and the
**actual** Unreal Engine 5.9 experimental MCP transport. Epic documents that
Unreal MCP runs **inside Unreal Editor** at
\`http://127.0.0.1:8000/mcp\` by default, with **Streamable HTTP or SSE**,
**not stdio**. Its server has **no authentication**, and its default tool
search mode returns only meta-tools rather than every editor tool.

Official reference:
https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor

## R0.3 deliberately does less than an editor agent

- Sends \`initialize\`, \`notifications/initialized\`, and paginated
  \`tools/list\` over the documented local HTTP transport.
- Supports bounded JSON and SSE JSON-RPC replies, session IDs, protocol
  version, and tool catalog pagination.
- Connects **only** to a literal \`127.0.0.1\` HTTP endpoint on port
  1024–65535, with the path exactly \`/mcp\`; forbids proxies,
  redirects, URL credentials, queries, remote hosts and HTTPS.
- Requires \`--approve-connect\` before a single connection. Defaults to
  no network activity, including when the local MCP server is running.
- Produces a metadata-only receipt with tool names and SHA256 digests of
  advertised \`inputSchema\`. No tool descriptions or raw schema content
  is promoted to instructions; nothing is executed based on metadata.
- Reuses R0.1 private audit log with STARTED/COMPLETED/FAILED entries.
- **Never invokes \`tools/call\` or \`call_tool\`**. No scene changes,
  project modifications, asset generation, editor commands, builds,
  Actions dispatch, merge or deployment.
- A passing mock-server test is **not evidence** that an actual Unreal
  installation connected, that the server is Unreal, or that an existing
  project is isolated.

## Pilot setup on a local Windows workstation

1. Create a disposable blank Unreal Engine 5.9 test project. Ensure all
   production UE projects are closed and any test content is backed up.
   Do **not** point at BTM, Race to Oblivion, or any other production
   project. This adapter itself cannot establish which editor project is
   running; human workstation isolation is still required.
2. In Unreal Editor, enable **Unreal MCP**, **All Toolsets** and the
   dependency **Toolset Registry** per Epic documentation; restart if
   prompted. Check **Editor Preferences → Agentic Workflows → Model
   Context Protocol**. Confirm the server binds to loopback only, and
   note the port (default 8000). Do not expose the editor MCP server to
   LAN, WAN or browser-based clients. Do not add external allowed origins.
3. Create a private audit directory outside your repository (for
   example \`C:\Users\<you>\mcp-audit\`) and keep receipts local.
4. Inspect available CLI options without opening a connection:

       python -m phistream.mcp_unreal_discovery --help

5. Run **discovery only**, explicitly authorizing one local connection:

       python -m phistream.mcp_unreal_discovery \
         --endpoint http://127.0.0.1:8000/mcp \
         --audit /your/private/mcp-audit/unreal.jsonl \
         --approve-connect

   The path in this example is POSIX-style for portability; supply a
   valid local Windows path when using PowerShell.

6. Review \`advertised_tools\`, \`input_schema_sha256\`, and the audit
   file privately. The client records
   \`LOCAL_CATALOG_OBSERVED_NOT_ATTESTED\`, not \`UNREAL_VERIFIED\`.

## What Epic's tool search changes

The stock Unreal plugin commonly lists just \`list_toolsets\`,
\`describe_toolsets\`, \`describe_tools\`, and \`call_tool\` in
\`tools/list\`. This adapter deliberately **does not call** these tools,
since MCP metadata alone cannot verify if a meta-tool invocation is
side-effect free. Therefore:

- The catalog is explicitly marked \`discovery_exhaustive=false\`.
- A meta-tool listed in the catalog is **not authorized** by its presence.
- Custom project toolsets and their eventual schemas require a second,
  separately reviewed qualification phase before execution.

## CI and what it proves

    python -m unittest tests.test_mcp_unreal_discovery -v
    python -m unittest discover -s tests -v

CI uses a local Python \`ThreadingHTTPServer\` fixture with synthetic
initialization, paginated tool catalogs, JSON replies, SSE replies,
malicious descriptions, invalid IDs, refused redirects and policy denials.
No Unreal installation or actual production project is exercised.
No external network calls are needed. Results only qualify local
protocol parsing and refusal behavior.

## Remaining gates before touching a scene

1. Inspect an **actual** UE 5.9 disposable project's tools using the
   read-only discovery CLI. Capture local receipts; do not publish
   sensitive schema or project data.
2. Pin/verify the editor executable and selected project identity.
   R0.3 cannot currently authenticate the server or project.
3. Determine which editor tools are genuinely read-only and which may
   mutate an editor, execute Python/C++, write disk, run tests or
   trigger external network operations. Treat meta-tools as privileged
   until independently tested.
4. Add a separately approved project-scoped \`tools/call\` path with
   schema validation, per-effect approval, and pre/post project-file
   fingerprints. Only disposable assets are eligible.
5. Reconcile outputs with tests, audit receipts and human observation
   before proposing integration with Agentora, PhiMadaPrime, BTM or
   Race to Oblivion.

**No auto-merge or unattended editor mutation.** The purpose of R0.3
is to validate Epic's actual transport and discover what the server
advertises without pretending that a local HTTP response proves
software safety.
