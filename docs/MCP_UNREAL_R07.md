# PhiStream MCP R0.7: actual Unreal field-test evidence intake

**Status: experimental, operator-driven, offline evidence review.**
No GitHub Actions runner has a real Unreal 5.9 editor session, and R0.7
does NOT claim actor creation, editor boot or full project recovery.
It prepares a reproducible evidence packet for a **local disposable project**
and keeps a human's report distinct from independent verification.

Epic's Unreal MCP documentation:
https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor

## What's implemented

\`phistream/mcp_unreal_field_qualification.py\` provides two CLI actions:

- \`template\`: create a **new** eight-question JSON checklist from an R0.5
  PointLight attempt plan. Every answer defaults to \`false\`. The program
  never overwrites an existing checklist or fabricates a successful field test.
- \`assess\`: independently re-read R0.6's byte-verified baseline, the
  original scratch-project file delta, the exact R0.5 attempt reservation,
  the optional R0.5 self-reported receipt, and a distinct recovered
  project directory. Compare **all monitored recovered files byte for
  byte** against the baseline. Capture SHA256/size/dimension metadata
  for four operator-provided screenshots. Check the operator's
  observations for consistent shape, exact planned actor label, a saved
  \`.umap\` delta, intact project anchors and a non-error R0.5 receipt.

This creates \`phistream.mcp-unreal.field-qualification.v0.7\` with
either:

- \`OPERATOR_REPORTED_FIELD_CHECKS_CONSISTENT_UNVERIFIED\`: all local
  conditions agree and the operator marked every check true. **Not
  independent proof**, not permission to run new tools.
- \`HOLD_FIELD_CHECKS_INCOMPLETE_OR_CONFLICTING\`: the field checklist is
  incomplete, the saved map has no observable disk delta, an R0.5 receipt
  is missing or reports an error, or the original project anchors differ.

Malformed, symlinked, missing or tampered required evidence is rejected
with a failure rather than a fabricated positive finding.

## Actual local workflow (Windows)

### 1. Before invoking any actor tool

Use a **new, disposable Unreal 5.9** project
\`C:\Scratch\PhiMCPSandbox369\`; close production Unreal projects.
Follow [R0.4](MCP_UNREAL_R04.md), [R0.5](MCP_UNREAL_R05.md)
and [R0.6](MCP_UNREAL_R06.md) to configure a loopback-only editor
MCP endpoint, create the disposable project marker, inspect **actual**
advertised tool names and schemas, and capture an offline baseline
**while the editor is closed**.

Do not invent an \`EditorToolset\` tool name or argument key.
If your installation does not advertise a compatible direct
PointLight-creation tool, stop; R0.5's mock example is not authorization
to use the general \`call_tool\` dispatcher.

### 2. Perform only the separately approved one-shot R0.5 actor attempt

While the scratch editor is open, use your operator-created R0.5 plan
with exact project, marker, instance, PID and tool-schema fingerprints.
The R0.5 \`*.attempt\` reservation is consumed once whether the call
succeeds, fails or times out. **Do not automatically retry.**

### 3. Capture manual observations from the original editor

Look at the actual World Outliner and the selected actor's Details panel
in the **original** disposable editor. Verify the intended
\`PhiMCPProbe_*\` PointLight, correct class, actual active project,
selected level and transform. Save the test level only if it is safe
and understood; R0.7 expects a saved \`.umap\` change compared with the
pre-attempt baseline.

Save these **PNG screenshots** in an existing private folder outside
the project and outside the recovery/baseline folders, using exactly:

- \`original-outliner.png\`: project/level context and actor visible
- \`original-actor-details.png\`: selected PointLight class, label, transform
- \`original-level-saved.png\`: saved original test-level context
- \`recovered-outliner.png\`: restored editor's World Outliner

The code checks PNG header dimensions, size, and bytes SHA256. It does
**not** inspect pixels or certify that screenshots are authentic.

### 4. Close the original editor and run R0.6 comparison/recovery

After manually observing the original editor and safely closing it,
use the R0.6 \`compare\`, \`review\`, and \`restore-copy\` commands.
Do not restore over the original project.

Open the restored copy in a **distinct Unreal editor session**, verify
that the scratch level actually loads and that the test actor is absent.
Take \`recovered-outliner.png\` from that separate restored copy.
Don't confuse identical project **names** with the same project path.

The recovered scratch directory for assessment might be:

    C:\Private\PhiRecoveryOutput\PhiMCPSandbox369

It must have the same exact bounded file inventory as the pre-edit
baseline; an R0.6 file match does **not** prove Unreal can open it.

### 5. Create an operator checklist (all false initially)

    python -m phistream.mcp_unreal_field_qualification template --plan "C:\Private\pointlight-plan.json" --output "C:\Private\field-observation.json"

Review the actual editor, saved map and recovered editor yourself;
set the eight boolean entries to \`true\` only if personally observed.
Do not use an AI agent to fill them based on the screenshots or the
R0.5 response. The expected actor label is copied from the plan and
cannot be changed without an assessment HOLD.

### 6. Assess everything offline

    python -m phistream.mcp_unreal_field_qualification assess --snapshot "C:\Private\PhiSnapshots\r06-before-light369" --project-root "C:\Scratch\PhiMCPSandbox369" --recovered-root "C:\Private\PhiRecoveryOutput\PhiMCPSandbox369" --plan "C:\Private\pointlight-plan.json" --intent-dir "C:\Private\mcp-intents" --receipt "C:\Private\pointlight-receipt.json" --operator-record "C:\Private\field-observation.json" --screenshots "C:\Private\PhiFieldScreenshots"

The packet is **metadata only**: SHA256 fingerprints, observed
boolean claims, change counts, restricted PNG dimensions and explicit
trust limitations. It does not launch Unreal, call MCP or run a model.
Keep it local until reviewed. The included screenshots are not
copied, uploaded or sent by the CLI.

### 7. Independent review (still a separate step)

A second human or reviewer can inspect the *actual* screenshots,
recorded hashes, local editor instance and recovered project. A
complete operator form does **not** authenticate that person, the
screenshots, the MCP server, the point-light state, or the actor's
provenance. No automatic gate changes
\`field_test_independently_qualified\` to \`true\`.

Only after genuine field evidence and separate acceptance should we
consider an R0.8 read-only actor-query adapter with the exact **observed
UE 5.9 tool schema**. Avoid exposing the unguarded Unreal editor MCP
server to LAN/WAN and never treat the local HTTP interface as
authenticated.

## Automated qualification

    python -m unittest tests.test_mcp_unreal_field_qualification -v
    python -m unittest discover -s tests -v

CI uses synthetic \`.umap\` bytes and synthetic PNG-like headers. It
does **not** start Unreal, decode real screenshots, open recovered maps,
validate editor process identity, prove a PointLight existed, or certify
product release. R0.7 does not write project files, merge PRs, dispatch
workflows or change MCP authorization.

## Known limitations

- A modified \`.umap\` file is *evidence of changed bytes*, not proof
  of which actor or map change caused it.
- The screenshot SHA256 binds the exact bytes offered by the operator,
  not the image's origin or meaning.
- The R0.5 result and human checklist are unsigned local self-reports.
- This is bounded monitoring of \`.uproject\`, marker, \`Content\`,
  \`Config\`, and \`Plugins\`, **not a complete Unreal project snapshot**.
- A malicious local process could fake all the observed evidence;
  this is not a cryptographic remote attestation system.
- All commands fail closed on malformed input and never grant execution,
  code-write, production, merge, deployment, or automatic retry authority.
