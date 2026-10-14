# ΦStream Web

Public, dark-mode React/Vite GitHub Pages companion for the **PhiStream experimental inference research engine**.

## Local development

From the repository root:

```sh
cd site
npm install
npm run dev
```

The build verifies browser-only evidence validators with Node's test runner, then validates and copies two committed **CPU synthetic fixture** receipts from `../benchmarks` into `public/benchmarks`. The UI does not collect data, connect to a GPU or execute models. It separately presents user-local experimental RTX 5070 research observations without claiming independent verification or general speedups.

## Deployment

The root `.github/workflows/pages.yml` builds the site and deploys `site/dist` using official GitHub Pages Actions on pushes to `main` and manual workflow runs. PRs build but do not publish. Base path is `/PhiStream/`. Published URL, when GitHub Pages is enabled: https://michaelwave369.github.io/PhiStream/

If GitHub Pages isn't enabled, repository Settings → Pages → **Build and deployment** → **GitHub Actions**. Repository admins must configure this when necessary; the deployment workflow cannot promise to enable Pages automatically.

The **capacity explorer is arithmetic only** (parameter count × storage bytes/parameter) and reserves an illustrative slice of device memory. It is not a GPU feasibility calculator. The older committed benchmark cards use synthetic CPU fixtures; the new real-model research section is explicitly separated as a user-reported local RTX 5070 experiment. No model files or private full CUDA receipts are shipped.

## PR #27 real-model research dashboard

The Research section presents the **user-local, offline-audited** seed-369 results for three frozen SmolLM2-135M-Instruct prompt cases (7, 12, and 24 tokens): 12/12 paired BATCHED wins, median paired decoder-time differences of 15.53%, 24.77%, and 39.68%. The 30 executed-trial parity comparisons recorded 960/960 greedy token matches and 8,590/8,590 sampled-logit matches. These counts are based on the user's prior local review, not a model run inside the site or repository and not independent provenance or statistical significance proof. The production default remains SINGLE. The cross-day 8-pair replication (seed 70421) is **planned only**, with no results claimed.

The local Evidence Explorer optionally reads either `PHISTREAM_THREE_CASE_PAIRED_AB_CENSUS_V1` or `PHISTREAM_PAIRED_REAL_PREFILL_AB_V1` JSON chosen by the viewer. JavaScript bounds file size, checks expected schema/ledger counts, pair order and trial statuses, and recomputes paired times for the latter. This is display-only structural validation, **not SHA256 authentication**. For real SHA-bound checks, use the offline Python protocols in [PR #25](../docs/PAIRED_REAL_PREFILL_AB.md) and [PR #26](../docs/THREE_CASE_AB_CENSUS.md). The site never uploads, saves, executes or transmits the contents of user-selected files.

Run `npm run test:evidence` in `site/` for the browser-validator regression suite. `npm run build` runs these tests automatically before Vite, so the GitHub Pages PR check fails for relevant regressions.

## Security

No secrets or user files are uploaded. GitHub Pages is static. The only runtime network request to data is for the public static benchmark JSON published with the site. Google Fonts are optional decorative resources, with local fallback fonts.

The **Local CUDA receipt viewer** under Evidence parses a user-selected V1 receipt entirely in-browser; nothing is sent to a server, stored remotely or published. It labels file-provided numbers explicitly. See [local device-validation instructions](../docs/DEVICE_VALIDATION.md).

The **Local Hardware Intelligence** reader loads either a `PHISTREAM_HARDWARE_SNAPSHOT_V1` or unapproved `PHISTREAM_RESIDENCY_ADVICE_V1` JSON file in-browser. No uploads or device probing occur in the website. Read [Hardware Advisor](../docs/HARDWARE_ADVISOR.md) for the Windows commands.

The **Transfer Laboratory** reader supports local `PHISTREAM_TRANSFER_LAB_V1` receipts, with bounded synthetic-file warm reads, RAM copy and optional synchronized pageable/pinned H2D timing. Files are read directly in the browser; no upload or backend execution. See [Transfer Laboratory](../docs/TRANSFER_LAB.md).

The **Paired Pipeline** reader accepts local `PHISTREAM_PIPELINE_LAB_V1` receipts, displaying complete synchronous pageable versus reusable pinned staging times, paired win counts and ratio. Browser-only, no backend. See [Paired Pipeline Lab](../docs/PIPELINE_LAB.md).

The **Reusable Buffer Laboratory** accepts a local `PHISTREAM_REUSE_LAB_V1` JSON receipt, comparing buffer allocation strategies across three synthetic sizes. Browser-only, no upload. See [Reuse Lab](../docs/REUSE_LAB.md).

The new **Direct Staging Laboratory** reader accepts local `PHISTREAM_DIRECT_READ_LAB_V1` receipts. It compares the legacy multi-copy path, direct pageable `file.readinto`, and direct pinned `file.readinto` across synthetic sizes; it never uploads files or runs device code. See [Direct Read Lab](../docs/DIRECT_READ_LAB.md).

The **Chunked Streaming** panel accepts local `PHISTREAM_CHUNK_PIPELINE_V1` receipts comparing single pinned and double pinned CUDA event-guarded chunk streams. The renderer reads local JSON in-browser without uploading it. See [Chunk Lab](../docs/CHUNK_LAB.md).

The browser-only **Overlap Intelligence** panel reads local `PHISTREAM_OVERLAP_LAB_V1` receipts and displays the persistent worker host timeline and conservative advisory without changing scheduler policy. See [Overlap Lab](../docs/OVERLAP_LAB.md).

The **Guarded Scheduler** reader accepts only local review receipts (`PHISTREAM_GUARDED_SCHEDULER_V1`) or explicitly executed synthetic CUDA reports (`PHISTREAM_GUARDED_SYNTHETIC_RUN_V1`). The website performs no approvals, CUDA execution, routing changes or uploads. See [Guarded Scheduler](../docs/GUARDED_SCHEDULER.md).

The offline **Scheduler Qualification** reader accepts a local `PHISTREAM_SCHEDULER_QUALIFICATION_V1` JSON report and shows every included guarded synthetic receipt, pass/hold evidence, randomized controls, and reasons for REJECTED/INCONCLUSIVE/QUALIFIED_SYNTHETIC. No files upload or GPU policy changes. See [Qualification Lab](../docs/QUALIFICATION_LAB.md).

The **Real-Checkpoint Bridge** browser-only JSON reader shows read-only preflight status, Llama architecture metadata, model execution/parity steps when explicitly provided, decoder measurements and separate source/license declaration warnings. It does not upload checkpoint weights or activate a scheduler. See [Real Model Bridge](../docs/REAL_MODEL_BRIDGE.md).

A local-only **HF Reference Trace** reader presents the separately generated 32-token external-runtime trace and checkpoint binding. Neither importing the JSON nor looking at a reference is proof PhiStream matches it. See [External Reference Lab](../docs/EXTERNAL_REFERENCE_LAB.md).

The read-only real-weight matrix panel accepts a local four-case review summary JSON. Its displayed status is **user-local and not authenticated by the browser**. The offline Python reviewer checks evidence binding; see [operator protocol](../docs/REAL_WEIGHT_MATRIX.md).

The offline weight staging audit accepts a user-local receipt and visualizes the SHA-bound four-case counter identity. It does **not** execute a model, authenticate the provided JSON, or claim achieved acceleration. See [method and limits](../docs/WEIGHT_STAGE_AUDIT.md).

## PR #28: operator-reported eight-pair replication

The React research page now separates the user's original seed-369 four-pair `spread_24` cohort (39.684% median paired decoder-time reduction) from the **operator-reported** October 13 eight-pair seed-70421 replication (40.5861%, 8/8 faster BATCHED pairs). These are the user's local terminal results, not independently audited or statistically certified results. The Python PR28 offline SHA-bound review is intentionally **pending until it is run on the operator's local receipt files**. The static website doesn't claim its existence changes the evidence status automatically.

A separate local-only upload reader accepts `PHISTREAM_REPLICATION_REVIEW_V1` JSON. JavaScript validates schema, original/replication comparison structure and combined counts; this is **not cryptographic SHA256 evidence verification**. No user files are uploaded, persisted or used for model execution. See [the operator audit procedure](../docs/REPLICATION_REVIEW.md).

The site build runs both the original frozen evidence-viewer tests and the replication-viewer regression tests. No new inference or experiment is triggered by this web update.
