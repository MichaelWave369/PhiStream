# PhiStream MCP R0.5: One-shot disposable Unreal PointLight pilot

**Experimental. No actual Unreal Editor connection, actor creation or rollback
has been field-tested by this PR or CI.** R0.5 adds a narrowly gated *option*
to attempt **one** PointLight placement in a disposable Unreal 5.9 project,
not production scene editing, agent authority, code generation, or autonomous
merge/deploy.

Official Epic reference:
https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor

## R0.5 build scope

- Reuses R0.4's operator-owned disposable project marker and project hash.
- Reads Unreal's reported initialize metadata at
  \`_meta["com.epicgames.unrealengine/instance"]\`: instance ID, process ID,
  project name/folder and engine version. Requires **Unreal 5.9**, an
  exact path/name match to \`PhiMCPSandbox*\`, and a pinned instance/PID.
  **These fields are not authenticated**; this is consistency evidence only.
- Uses the **direct, fully qualified MCP tool name** in the default
  \`tools/call\` protocol, not Epic's broad \`call_tool\` meta-tool.
  Therefore the disposable editor's **Enable Tool Search must be off**,
  and the exact actor-creation tool must be visible in \`tools/list\`.
- Narrow implementation recognizes only the name pattern
  \`EditorToolset.<Toolset>.SpawnActor\` or
  \`EditorToolset.<Toolset>.SpawnActorFromClass\`, pinned to its observed
  \`inputSchema\` SHA256. These names are **qualification candidates**, not
  guaranteed names in a particular UE 5.9 build. A different real tool
  name requires code review before this gate can support it.
- Allows exactly a PointLight class, one \`PhiMCPProbe_*\` actor name/label,
  and optionally small bounded XYZ position and rotation arguments.
  No generic class spawn, arbitrary package path, Blueprint, Python,
  shell command, asset import, save, delete or deployment.
  Argument keys are intentionally limited and **may not match** a real
  build's required schema. The pilot must refuse incompatible schemas,
  not silently loosen the controls.
- Requires separate operator connection/write CLI flags **and** an exact
  one-shot confirmation string. These aren't authenticated human identity.
- Uses an immutable, locally reserved \`*.attempt\` record, created with
  exclusive file creation before any network activity. A crash, timeout,
  server error or unknown outcome **consumes the intent**. Never
  automatically retry or assume nothing happened.
- Hashes bounded \`.uproject\`, marker, \`Config\`, \`Content\`, and \`Plugins\`
  file bytes before and after. This is not an in-memory scene diff and
  cannot detect changes outside these paths, so manual editor acceptance
  and rollback remain required.

## Field test protocol on your Windows machine

**Preconditions:** R0.4 PR merged; Epic UE 5.9 installed locally. The running
editor must be your independently inspected, disposable
\`PhiMCPSandbox369\` project, **not BTM, Race to Oblivion, or any production
project**. Close other editors, disable untrusted plugins, and back up
the entire disposable project before testing. Unreal MCP has **no
authentication** and should remain loopback only.

### 1. Preflight disposable project

Create a blank project and the marker following
[R0.4 documentation](MCP_UNREAL_R04.md):

    python -m phistream.mcp_unreal_qualification preflight --project-root "C:\Projects\PhiMCPSandbox369"

The marker file's contents and SHA must match its \`.uproject\`. No marker
is auto-generated.

### 2. Observe the reported editor instance, without tools

    python -m phistream.mcp_unreal_actor_pilot identity --project-root "C:\Projects\PhiMCPSandbox369" --approve-connect

Record \`instance.instance_id\` and \`instance.process_id\`. They are
**server-reported**, not security proof. If identity keys don't match the
installed UE build, the pilot must hold for review.

### 3. Observe a *direct* actor-creation tool

In the disposable editor's **Editor Preferences > Agentic Workflows >
Model Context Protocol**, turn **Enable Tool Search** off, reconnect the
client, and run R0.3 discovery:

    python -m phistream.mcp_unreal_discovery --audit "C:\Private\unreal-discovery.jsonl" --approve-connect

Look for one exact actor-tool name matching the narrow allowlist above and
record its \`input_schema_sha256\`. No actor edit is performed by discovery.
If no such compatible tool exists, **STOP**. Do not invent a name or expose
the generic \`call_tool\` dispatcher. Inspect the tool using Epic's
official MCP Inspector and qualify a new supported adapter separately.

### 4. Construct an explicit JSON attempt plan

Write the following to \`C:\Private\pointlight-plan.json\` **by hand**.
Replace placeholders with *actual* observations from steps 1–3; adapt
\`arguments\` only if its keys are exactly compatible with the observed
schema and R0.5's tight allowlist.

    {
      "schema": "phistream.mcp-unreal.actor-pilot.v0.5",
      "project_sha256": "<64 lowercase hex of .uproject>",
      "marker_sha256": "<64 lowercase hex of PHI_MCP_DISPOSABLE.json>",
      "expected_instance_id": "<observed server-reported instance ID>",
      "expected_process_id": 12345,
      "tool_name": "EditorToolset.ActorToolset.SpawnActor",
      "tool_schema_sha256": "<64 lowercase hex from R0.3>",
      "arguments": {
        "actor_class": "/Script/Engine.PointLight",
        "actor_label": "PhiMCPProbe_369"
      }
    }

**Example only:** The \`tool_name\` and \`arguments\` above were tested
against a **mock**, not confirmed against an actual Unreal installation.
If the real schema differs, hold rather than guessing a compatible payload.

### 5. Approve one attempt, with no automatic retry

Create an existing **private directory outside the project** for your
one-shot attempt records, such as \`C:\Private\mcp-intents\`. Then run a
single command in PowerShell (adapt local paths):

    python -m phistream.mcp_unreal_actor_pilot attempt --project-root "C:\Projects\PhiMCPSandbox369" --plan "C:\Private\pointlight-plan.json" --audit "C:\Private\unreal-actor-audit.jsonl" --intent-dir "C:\Private\mcp-intents" --approve-connect --approve-write --confirm ONE_DISPOSABLE_POINTLIGHT_ONE_ATTEMPT_NO_AUTO_RETRY

Any failure or unknown outcome after the intent record is reserved means:
**do not rerun the same plan**. Inspect the editor manually first.
One successful JSON-RPC response is **not proof** an actor exists.

### 6. Independently verify and reset

In the disposable editor, look for exactly one new PointLight with a
\`PhiMCPProbe_\` label. Check the selected map, transform, World Outliner,
asset dirty state and Content Browser, then compare the restored
disposable snapshot after closing without saving or reverting the copy.
Keep original and edited project files separate. Do **not** claim automated
rollback; the R0.5 CLI performs none.

## CI and explicit limits

    python -m unittest tests.test_mcp_unreal_actor_pilot -v
    python -m unittest discover -s tests -v

Tests use injected MCP clients, controlled local files and synthetic
server-reported identity. No actual UE instance, editor write, saved
map, real GPU or game project is used.

Even after green CI the runner is **not production-safe**: only a
disposable scratch environment is eligible. No authenticated host/process
binding, guarantee of tool behavior, file-wide change control, verified
actor existence, or automatic rollback. A self-reported tool response
remains untrusted. The exact argument names for a real actor creation tool
must be verified in a disposable editor before first use.
