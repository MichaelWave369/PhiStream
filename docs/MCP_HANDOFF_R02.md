# MCP R0.2: evidence handoffs without privilege escalation

**Status:** experimental, **offline / review-only**. Extends
[MCP R0.1](MCP_GATEWAY_R01.md) without changing its command or permission
checks. Imports *existing* public-only evidence contracts, not a new
cross-repo job queue:

- **Agentora R9** \`MissionControl.status()\`: source schema
  \`agentora.fleet.mission-control.v0.9\`, public-only mission status. Only
  \`QUEUED\` review missions are accepted, and the exact
  operator-selected 40-hex source SHA must match.
- **PhiMadaPrime R31**: exported structured
  \`phimada.agentora-r1-chronicle-correlation.v0.31\` result, selected
  tested-head or verified-merge SHA, tied to the *known Agentora repo*.
  R31's own implementation correlates GitHub records, but **this offline
  copy is not independently authenticated** by PhiStream.

Both imports give the same schema:
\`phistream.mcp-handoff.review.v0.2\`.
They contain source/manifest SHA256 fingerprints and a proposed
MCP server/tool that is already explicitly named in the operator
manifest. There is no tool call and no subprocess launch.

## CLI example (Agentora R9)

Run Agentora's existing \`status\` command using its own documentation
and private databases; save the returned *public-only* JSON snapshot.
Pick a mission whose R9 status is \`QUEUED\`.

    python -m phistream.mcp_handoff \
      --source /private/agentora-status.json \
      --source-kind agentora-r9 \
      --mission-id test-001 \
      --manifest /private/mcp-manifest.json \
      --server local-editor \
      --tool inspect_scene \
      --expected-sha 0123456789012345678901234567890123456789abcdef

The SHA in this example is a placeholder; supply the actual exact
source commit from Agentora's status.

## CLI example (PhiMadaPrime R31)

The R31 browser includes a **text** report for human reading. This
bridge needs a local JSON serialization of the structured R31
correlation **object**, not the copied prose report:

    python -m phistream.mcp_handoff \
      --source /private/phimada-r31.json \
      --source-kind phimada-r31 \
      --manifest /private/mcp-manifest.json \
      --server local-editor \
      --tool inspect_scene \
      --expected-sha 0123456789012345678901234567890123456789abcdef

The expected SHA must equal the R31 \`selectedSha\` and agree with
either \`testedSha\` or \`mergeSha\` as required by the relation.
The imported record is useful context, **not a fresh GitHub verification**.

## What the handoff DOES NOT do

- Does not invoke the MCP gateway, run MCP tools, start processes, or call
  GitHub APIs.
- Does not issue Agentora leases, modify missions, trigger Actions jobs,
  push branches, write code, merge, deploy, or perform model inference.
- Does not accept a mission receipt as an execution approval. Even an
  operator-declared \`write\` tool returns
  \`code_write_authorized=false\`.
- Does not claim agent identity, verified product acceptance, current
  GitHub HEAD, server provenance, or tool effects.
- Does not elevate PhiMadaPrime's R31 corroborated GitHub history into
  an authenticated cross-system instruction.
- Does not make copied source payloads trustworthy: the SHA256 values
  protect content binding only, not authorship or freshness.

The MCP gateway continues to require operator launch, execute, and
where applicable write approvals **separately**. A future adapter can
present this review packet to an operator UI, but must do fresh
provider/source verification and approval at effect time.

## Qualification

    python -m phistream.mcp_handoff --help
    python -m unittest tests.test_mcp_handoff -v
    python -m unittest discover -s tests -v

The tests use synthetic JSON, deliberately forged approvals, wrong SHA,
wrong state, unknown tool, duplicate mission, spoofed provenance and
malformed files. They do not involve real remote providers, GitHub runs,
MCP servers or devices. A CI pass only qualifies the offline validation
contract and does not attest to real integration execution.

Next: executable provenance (pinned SHA256 and signature where practical),
revocable policy review, read-only live status rechecks, and a disposable
sandbox editor MCP pilot. CI and independent human review must precede
any future code-writing worker.
