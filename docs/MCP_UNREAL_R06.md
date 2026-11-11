# PhiStream MCP R0.6: offline actor-attempt evidence and scratch recovery

Status: **experimental, offline. No real Unreal Editor used in CI.**

R0.6 builds on R0.5's one-shot actor pilot. It does not use the MCP
connection at all: it independently inventories a limited selection of
project files, correlates local R0.5 intent/receipt records, and proves
that a copied **subset** of an operator-created baseline can be reproduced
in a separate recovery directory.

### Scope, not mythology

**What R0.6 can demonstrate:** exact SHA256 parity for monitored
\`.uproject\`, disposable marker, and files under \`Content\`, \`Config\`,
and \`Plugins\`, within strict size/count limits; added, removed, and
modified files after an actor attempt; whether a saved R0.5 *self-reported*
receipt and one-shot reservation match the approved attempt plan; whether a
new recovery copy matches the captured baseline byte for byte.

**What it cannot demonstrate:** actor creation, World Outliner presence,
unsaved level state, which editor process authored a change, complete
Unreal project recovery, restoration of \`Saved\`, \`Binaries\`,
\`Intermediate\`, derived cache or external source-control state, editor
launchability, or automated rollback of an already-open editor. Its
comparisons do not authenticate the operating system or a local
attacker. No agent receives write, merge or production authority.

The Epic Unreal MCP server is experimental and has **no authentication**:
https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor

### Operator checklist: capture, experiment, compare, restore

All paths below are examples for a **Windows disposable scratch project**.
Make the folders with ordinary filesystem controls in advance.
Keep \`PhiMCPSandbox369\` in a separate test location, all production
Unreal projects closed, and a **separate full backup** of the scratch
project. Shut down the scratch editor before the baseline copy to avoid
partially captured or unsaved assets. R0.6 cannot enforce that shutdown.

#### Step 1. Capture a pre-edit baseline (no MCP connection)

Create \`C:\Private\PhiSnapshots\`, not inside the project. In PowerShell:

    python -m phistream.mcp_unreal_recovery capture --project-root "C:\Scratch\PhiMCPSandbox369" --snapshot-parent "C:\Private\PhiSnapshots" --name r06-before-light369

A complete snapshot contains
\`C:\Private\PhiSnapshots\r06-before-light369\baseline.json\` and the
copied, bounded files inside \`payload\PhiMCPSandbox369\`. An incomplete
capture leaves its partial folder behind and **never** marks it complete;
inspect and choose a new snapshot name rather than overwrite it.

The \`baseline.json\` is a local, operator-controlled manifest, **not
a signed attestation**. A captured file can be absent from this subset,
so do not treat this as your sole Unreal backup.

#### Step 2. Run R0.5 once

Restart your disposable editor, use R0.5's **actual observed** MCP
instance, tool name, schema SHA256 and fixed \`PhiMCPProbe_*\` PointLight
arguments. Review the complete
[MCP Unreal R0.5 guide](MCP_UNREAL_R05.md).
Save R0.5's JSON output **privately** to a local
\`C:\Private\pointlight-receipt.json\` file. This must be an exact
uncut JSON object, not a chat summary.

If a write times out or otherwise has an uncertain outcome, the
exclusive R0.5 \`*.attempt\` reservation is still consumed. Do not
repeat the same attempt or substitute a different intent to work around
uncertainty.

#### Step 3. Compare current files with the prior baseline

After closing Unreal (do not save or discard unsaved edits until you
have independently inspected the editor state):

    python -m phistream.mcp_unreal_recovery compare --snapshot "C:\Private\PhiSnapshots\r06-before-light369" --project-root "C:\Scratch\PhiMCPSandbox369"

The result lists relative paths (up to 80) and counts for added,
missing, and modified monitored files. It **always** reports
\`actor_existence_verified=false\`; a change in \`.umap\` bytes alone
doesn't prove the intended actor was placed.

#### Step 4. Correlate the R0.5 intent and optional self-reported receipt

    python -m phistream.mcp_unreal_recovery review --snapshot "C:\Private\PhiSnapshots\r06-before-light369" --project-root "C:\Scratch\PhiMCPSandbox369" --plan "C:\Private\pointlight-plan.json" --intent-dir "C:\Private\mcp-intents" --receipt "C:\Private\pointlight-receipt.json"

The exact source project directory, approved R0.5 plan, marker hash,
one-shot on-disk reservation and saved R0.5 response must match.
Without a response receipt, omit \`--receipt\`, but the result remains
**HOLD**. Receipt content and file deltas never become editor
attestation, actor proof or auto-retry permission.

#### Step 5. Restore into a different scratch location, not the original

Create an **empty** \`C:\Private\PhiRecoveryOutput\` directory outside
both snapshot and original project paths. Then:

    python -m phistream.mcp_unreal_recovery restore-copy --snapshot "C:\Private\PhiSnapshots\r06-before-light369" --destination-parent "C:\Private\PhiRecoveryOutput"

The program creates a **new**
\`C:\Private\PhiRecoveryOutput\PhiMCPSandbox369\` and independently
rehashes every copied monitored file against the baseline. It refuses
an existing target, invalid or tampered baseline, links in monitored
paths, oversized projects and overlapping parent locations.
It never deletes/replaces the original project or claims to roll back
an open editor.

**Manually** open the restored scratch copy in a separate editor process
and validate it loads, the expected map and assets are present, and the
test PointLight is absent in the restored state. Only that explicit
real-editor observation can support a claim about the level contents.
A restored file hash alone cannot prove this.

### Testing and failure semantics

    python -m unittest tests.test_mcp_unreal_recovery -v
    python -m unittest discover -s tests -v

The tests exercise synthetic bytes with \`.umap\`/project extensions,
not UE-formatted levels. They cover complete snapshots, exact byte
parity, changes, deletion, tamper, symlinks, limits, separate-copy
restore, no-overwrite, forged R0.5 receipts, unknown outcomes, and
production-named project denial.

- If any snapshot verification fails, no restore destination is created.
- On a partial copy failure, a new destination may be left incomplete.
  No tool silently deletes, retries or overwrites it.
- R0.5's original private one-shot reservation remains intact.
- Review results never authorize further editing, sending a second
  write, auto-merging, deploying, or treating a source-controlled
  project as disposable.

### R0.7 qualification gate

Perform the first **real** disposable Unreal Editor smoke test on a
local Windows workstation, record genuine tool catalog + schemas,
use an operator-driven actor attempt, inspect the actual map and
World Outliner, and independently open the R0.6 recovered copy.
Then, and only then, consider a strictly allowlisted read-only
editor actor observation adapter, pinned to its actually observed
tool schema. Human acceptance stays distinct from CI status.
