# PhiStream MCP Gateway R0.1: governed tool control plane

Status: **experimental, local stdio only, denied by default**. This is a thin
integration boundary, NOT an autonomous fleet, an Unreal integration, a
deployment mechanism, a trusted tool classifier, or evidence of end-to-end
workflow safety. GA108/Vessie retain routing authority; CLAVIS proposals
remain proposals, never an approval token.

## Capabilities already implemented

1. Load an **operator-authored**, schema-checked manifest containing absolute
   executable paths, bounded argv, explicit tool allowlists and human-assigned
   read/write classifications.
2. Launch a **single, explicitly approved local stdio MCP process** (no shell),
   negotiate JSON-RPC initialization and retrieve a bounded, paginated tools
   catalog. A server can advertise a tool without being granted permission.
3. Refuse execution unless **both** launch and invocation are approved. Tools
   classified as write also require a distinct write approval. No wildcards.
4. Check the tool is both allowlisted **and actually advertised** before calling
   it. Timeouts, response caps and protocol validation fail closed.
5. Write restrictive append-only **metadata** JSONL: operation, status, tool,
   argument hash and result hash. The ledger does not contain raw arguments,
   results, environment-variable values, stdout or model prompts.
6. Print only a hashed result receipt unless an operator explicitly requests
   raw output. Raw MCP tool output is **untrusted data**, not instructions.

## Example manifest

Save as an operator-controlled file OUTSIDE the repository. Replace command
and arguments with an installed **trusted** local MCP server executable.

    {
      "schema": "phistream.mcp-gateway.manifest.v0.1",
      "servers": {
        "local-editor": {
          "command": "/absolute/path/to/trusted/mcp-server",
          "args": ["--stdio"],
          "allowed_tools": {
            "inspect_scene": "read",
            "create_blueprint": "write"
          },
          "timeout_seconds": 15,
          "pass_env": []
        }
      }
    }

The example is a **policy shape**, not a claim that these specific tools exist
in Unreal or that Unreal MCP supports a particular binary path. Before real
use, inspect an actual server's advertised names, trust boundaries and
effects. A tool labeled "read" here is only an operator assertion. Tools that
can mutate state, trigger code execution, access networks, deploy, write to
repositories, run workflows, or send messages should be treated as **write**
and not as harmless inspections. V1 has **no automatic merge/deploy tools**.

### CLI

The audit directory must already exist. Keep the manifest, audit file,
arguments and all credentials outside the git checkout:

    python -m phistream.mcp_gateway \
      --manifest /private/mcp-manifest.json \
      --audit /private/mcp-audit.jsonl \
      catalog --server local-editor --approve-launch

Discovery starts the configured server. Starting a server may itself have side
effects; inspect the local executable and configuration first. Without
\`--approve-launch\`, even discovery is rejected **before spawning**.

A read-only classified tool requires both flags:

    python -m phistream.mcp_gateway \
      --manifest /private/mcp-manifest.json \
      --audit /private/mcp-audit.jsonl \
      call --server local-editor --tool inspect_scene \
      --arguments-file /private/scene-args.json \
      --approve-launch --approve-execute

A write-classified tool additionally requires \`--approve-write\`. Output is
a SHA256 receipt by default. The optional \`--show-result\` exposes the entire
untrusted server response, so never use that flag in shared logs or CI.

A separate local command works without any external MCP server:

    python -m phistream.mcp_gateway --help
    python -m unittest tests.test_mcp_gateway -v

The CI suite auto-discovers \`tests/test_mcp_gateway.py\`, including a tiny
controlled Python stdio MCP fixture; it launches no editor, never downloads
models, and performs no real external writes. A green CI result establishes
these local contract tests only, NOT production tool safety or permission
correctness for third-party servers.

## Security boundaries and limitations

- The gateway **cannot authenticate** a third-party server or reliably
  determine the side effects of a tool. Only register trusted local executables
  and classify each tool manually. It does not replace OS sandboxing.
- \`pass_env\` is an explicit list of environment-variable names forwarded
  to the subprocess, not literal secret values. The server can access these.
  A small OS/PATH environment is still forwarded for runtime compatibility.
- Server stdout must conform to MCP newline-delimited JSON-RPC. Logs and
  arbitrary plaintext written to stdout will cause a protocol failure.
- MCP **HTTP/SSE transports, persistent sessions, resources, prompts, sampling,
  elicitation, OAuth, remote services, agent-to-agent delegation and dynamic
  credential provisioning are not implemented** in this release.
- A CLI boolean approval does not prove a human physically clicked a button;
  higher-level agent callers must NOT self-authorize. Human approval and OS
  isolation are responsibilities of the trusted invoking application.
- The audit records a *local observation*, not a tamper-proof attestation. Keep
  it private and preserve/ship it to an independently protected log if used
  for sensitive operations.
- No child process starts if the pre-execution audit write fails, and failed
  calls are recorded. Unexpected host termination can leave a STARTED record
  without a matching completion, which is not proof of execution success.
- Per-call timeouts bound individual JSON-RPC waits. This is not a global
  lifecycle timeout; server startup, tool listing and a tool call each have
  separate limits.
- MCP tool descriptions can contain prompt injection. Treat catalogs and
  tool results as hostile text; do not feed either into a model as authority.

## Proposed next qualification gates

**R0.2: Capability review.** Record signed/verified executable identity,
manifest hashing, operator policy provenance and a negative-test matrix for
malicious/misclassified servers. Integrate permission-scoped adapters for
Agentora and PhiMadaPrime without bypassing this central gate.

**R0.3: Real adapter pilot.** Use a sandbox Unreal/other editor MCP server,
read-only discovery first, then an approved disposable test project. Record
observed tool schema, return types, fixture parity and side effects. Do not
wire to BTM/Race to Oblivion production saves until verified.

**R0.4: CI-aware change promotion.** Stage generated changes only on isolated
feature branches; bind approval to immutable commit SHAs; require test,
security and human-review receipts before merge. GitHub Actions checks must
be observed and verified, not guessed from agent text. Autonomous merge is
outside R0.1.

This modular path preserves PhiStream's existing benchmark-first,
fail-closed design instead of silently promoting Nine Mind/CLAVIS planning
to runtime autonomy.
