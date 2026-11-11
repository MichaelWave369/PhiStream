# PhiStream MCP R0.8: capture actual Unreal tool schemas before R0.5 write

**Experimental, workstation-side discovery only. No Unreal Editor is run
inside GitHub Actions.**

This rung addresses the biggest gap left by R0.5: the PointLight tool name
and argument keys used in unit tests were mock examples, **not observations**
from a real Unreal 5.9 tool catalog. R0.8 exports the actually advertised
tools into a **private JSON file** for operator inspection without invoking
any tool, spawning an actor, or enabling automated writes.

Epic official documentation:
https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor

## Setup (Windows workstation)

1. Create a fresh blank **PhiMCPSandbox369** Unreal Engine 5.9 project.
   Never use BTM, Race to Oblivion, or another production project for this.
   Follow [R0.4](MCP_UNREAL_R04.md) to create the
   \`PHI_MCP_DISPOSABLE.json\` file by hand, with the SHA256 of the
   local \`.uproject\`. The marker is a self-assertion, not cryptographic
   editor process authentication.
2. In Unreal Editor enable **Unreal MCP** and **All Toolsets** (Toolset
   Registry dependency), keeping the local server bound only to
   \`http://127.0.0.1:8000/mcp\` or another localhost port you explicitly
   choose. Do not add network origins or port-forward the MCP server.
3. In **Edit > Editor Preferences > Agentic Workflows >
   Model Context Protocol**, turn **Enable Tool Search OFF**. The default
   setting exposes only four meta-tools. Turning it off causes
   \`tools/list\` to list individual tools and schemas instead. If your
   tool catalog is too large for our R0.3 512-tool limit, the client
   will fail closed: use the official MCP Inspector locally to qualify
   schema discovery, then extend the bounded paging policy after review.
4. Create a private directory **outside the project** and keep it
   separate from your Git checkout, for example \`C:\Private\PhiMCP\`.
   The local machine is the only target. This ChatGPT session cannot
   connect to your workstation's \`127.0.0.1\`.
5. Run in PowerShell on the machine running Unreal (one line):

       python -m phistream.mcp_unreal_schema_capture --project-root "C:\Scratch\PhiMCPSandbox369" --private-output "C:\Private\PhiMCP\unreal-schema-r08.json" --audit "C:\Private\PhiMCP\unreal-schema-audit.jsonl" --approve-connect --approve-export

   Or replace the endpoint using \`--endpoint http://127.0.0.1:8123/mcp\`
   if you explicitly changed the local port.

This sends **only** \`initialize\`, \`notifications/initialized\`, and
paginated \`tools/list\`. There is no \`tools/call\`, \`call_tool\`, or
filesystem write into the project.

## What to inspect in private output

\`unreal-schema-r08.json\` contains:

- The Unreal MCP instance ID, reported process ID, engine version,
  project name, project and marker SHA256. These are **server-reported**
  and are not an authenticated identity binding.
- A sorted array \`advertised_tool_hashes\` with **every** advertised tool
  name and SHA256 of its input schema.
- Up to 16 \`selected_input_schemas_untrusted\` entries for names
  containing \`Actor\` or \`Spawn\`, including the full **untrusted**
  \`inputSchema\` of each. Large catalogs are truncated deliberately
  for this *subset*; the complete name/hash list is retained.
- A \`tool_search_mode_apparent\` indicator. If only the four meta-tools
  are present, the tool-search setting is apparently still on. Turn it
  off **in the disposable editor** and reconnect. No generic dispatcher
  calls are used.

To inspect one additional exact advertised tool schema, repeat with a
**new** filename and \`--exact-tool "Exact.Tool.Name.As.Advertised"\`.
The code never guesses argument names or executes that tool.

The private input schema may contain third-party descriptions or
prompt-injection strings. Treat the entire JSON file as **untrusted data**.
Do not hand it to an autonomous model as instructions, publish it to
a public GitHub PR, or paste secret project data into logs.

## Qualification decision

Compare the **observed** tool name, \`input_schema_sha256\`, allowed
input properties, required properties, class values and actor-label
behavior with [R0.5's one-shot policy](MCP_UNREAL_R05.md).

- If a real directly advertised tool matches the narrow R0.5 pattern,
  and the actual arguments exactly fit its JSON schema, a human can
  prepare the R0.5 disposable-project plan for a **separately approved**
  write.
- If the installed UE build exposes different names or required
  parameters, **STOP** and implement a specific reviewed adapter
  revision. Do not broaden R0.5 to arbitrary tools or use \`call_tool\`.
- If the only response is the default tool-search meta-tools, turn
  tool search off and recapture. No editor writes are authorized.

Even a complete observed schema does **not** prove the server is the
real Unreal process or that the tool behaves as advertised. The real
PointLight field test and R0.7 independent evidence review are still
pending.

## CI validation

    python -m unittest tests.test_mcp_unreal_schema_capture -v
    python -m unittest discover -s tests -v

These tests use local synthetic tool catalogs and server-reported metadata.
The established R0.3 tests already cover HTTP/SSE transport. R0.8 tests
focus on explicit approval, exact metadata capture, output path
confinement, exclusive creation/no overwrite, oversized schemas, unsafe
tool descriptors, schema hashes, tool-search mode detection, and
**never issuing a tool call**.

**No Unreal Editor installed in CI. No scene modified. No full editor
attestation, automatic actor proof, auto-retry, auto-merge, or production
authorization.**
