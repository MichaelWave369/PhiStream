# PhiStream MCP R0.4: Unreal metadata qualification in a disposable project

**Status: experimental. No production authority. No actual Unreal Editor field
test has been performed in CI.**

This builds on [R0.3 discovery](MCP_UNREAL_R03.md). Unreal 5.9 defaults to
tool-search mode, where \`tools/list\` exposes the four meta-tools
\`list_toolsets\`, \`describe_toolsets\`, \`describe_tools\`, and \`call_tool\`.
R0.4 deliberately allows **only the first three** and rejects the generic
\`call_tool\` dispatcher and ALL direct editor-modifying tools.

Official documentation:
https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor

## What is new

- A read-only local project preflight that checks one uniquely named
  \`PhiMCPSandbox*.uproject\` in a same-named root folder.
- An **operator-authored** \`PHI_MCP_DISPOSABLE.json\` marker that identifies
  the exact local \`.uproject\` file by SHA256. It is a precaution, not
  cryptographic attestation of which project Unreal currently has open.
- A metadata-only \`tools/call\` path for three fixed tool names, with two
  explicit CLI approvals: \`--approve-connect --approve-inspect\`.
- A separate operator-supplied SHA256 pin for that tool's **observed**
  \`inputSchema\` from R0.3. Changing schemas refuse execution.
- A private audit trail recording action status and content fingerprints,
  not returned tool descriptions, project paths, or raw responses.
- A post-call check that the local \`.uproject\` and marker still match
  their original hashes. This does **not** prove no changes to Unreal's
  in-memory scene, other assets, cache, project config or outside paths.

## Workstation instructions

1. Install/run UE 5.9 using Epic's documentation. Enable **Unreal MCP** and
   **All Toolsets** (dependency: Toolset Registry).
2. Make a **new blank disposable** project named \`PhiMCPSandbox369\`
   in a directory with the same name. Do not reuse BTM, Race to Oblivion,
   or other production projects. Keep production editors closed while testing.
   The application cannot authenticate the running editor/project.
3. Find the project's \`PhiMCPSandbox369.uproject\` SHA256. In PowerShell:

       Get-FileHash 'C:\Path\PhiMCPSandbox369\PhiMCPSandbox369.uproject' -Algorithm SHA256

4. Add the following JSON **by hand** to
   \`PhiMCPSandbox369\PHI_MCP_DISPOSABLE.json\`, substituting the exact digest:

       {
         "schema": "phistream.mcp-unreal.disposable-project.v0.4",
         "project_filename": "PhiMCPSandbox369.uproject",
         "project_sha256": "<lowercase 64-hex SHA256 from Get-FileHash>",
         "disposable": true
       }

   This marker is not written automatically by any agent.
5. Perform the **offline** local preflight:

       python -m phistream.mcp_unreal_qualification preflight --project-root 'C:\Path\PhiMCPSandbox369'

   It will say \`LOCAL_MARKER_MATCHED_NOT_EDITOR_AUTHENTICATED\`.
6. Run the R0.3 discovery command in
   [MCP_UNREAL_R03.md](MCP_UNREAL_R03.md), selecting the local endpoint
   documented by Epic (default \`http://127.0.0.1:8000/mcp\`).
   Inspect the hash recorded for the \`list_toolsets\` or
   \`describe_tools\` input schema in the R0.3 output.
7. Create \`C:\Private\mcp-args.json\` with **only** the intended metadata
   parameters. For an \`list_toolsets\` invocation, use \`{}\`.
   For \`describe_tools\`, consult the exact schema from the live local
   catalog; do not guess the parameter names. The application does not
   convert or synthesize those parameters.
8. After personally inspecting the marker, open editor, tool name, argument
   file, tool-schema digest and local MCP server settings, run:

       python -m phistream.mcp_unreal_qualification inspect ^
         --project-root "C:\Path\PhiMCPSandbox369" ^
         --endpoint http://127.0.0.1:8000/mcp ^
         --audit "C:\Private\mcp-audit.jsonl" ^
         --tool list_toolsets ^
         --arguments-file "C:\Private\mcp-args.json" ^
         --expected-schema-sha256 <64-hex digest recorded by R0.3> ^
         --approve-connect --approve-inspect

   The example above uses cmd.exe continuation \`^\`. In PowerShell use
   one long line or PowerShell's backtick. The audit directory must
   already exist. The tool response is fingerprinted and never printed.
   **This is a metadata query, not a scene edit.**

## Security/verification limits

- Epic's plugin has **no authentication**. A local HTTP service answering
  at the configured URL is not proof that the service is Unreal, or that
  the intended disposable project is the one open in the editor.
- An operator-editable marker is not a secure sandbox. A compromised local
  process can fake MCP responses and perform arbitrary side effects.
- A tool named \`describe_tools\` can behave maliciously if the local
  server is compromised. The pinned input-schema hash is not an attested
  tool implementation. Inspection requires separate human approval.
- Other editor state or assets may mutate even if the \`.uproject\`
  identity anchor is unchanged. The receipt therefore explicitly says
  \`editor_scene_changes_verified_absent=false\`.
- No R0.4 API can invoke the generic \`call_tool\` meta-tool, spawn an actor,
  execute arbitrary Python, write code, publish a PR, dispatch workflows,
  merge changes or deploy.
- CLI approval flags are **not proof of a physical human approval**.
  Agents must not supply them automatically.
- Never expose Unreal MCP beyond loopback, add broad allowed origins,
  forward private MCP data into public CI, or operate a production project.

## Verification scope

The test suite uses an isolated mock project and both injected protocol
clients and an actual local \`ThreadingHTTPServer\` fixture. Nothing installs
Unreal, downloads assets or opens a real editor. Run:

    python -m unittest tests.test_mcp_unreal_qualification -v
    python -m unittest discover -s tests -v

The **next gate**, once this code passes CI and an operator produces
a clean local R0.4 observation, is a disposable project **actor creation**
pilot. Before any write, add real editor project/process identity checks,
per-asset pre/post inventory, isolated one-shot edit scope, explicit
write approval and a rollback strategy. An actor appearing in a throwaway
map is NOT proof of safe autonomous game development.
