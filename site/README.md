# ΦStream Web

Public, dark-mode React/Vite GitHub Pages companion for the **PhiStream experimental inference research engine**.

## Local development

From the repository root:

```sh
cd site
npm install
npm run dev
```

The development and build scripts validate and copy the two committed **CPU synthetic fixture** receipts from `../benchmarks` into `public/benchmarks`. The UI does not collect data, connect to a GPU, execute models, or claim benchmark results for consumer hardware.

## Deployment

The root `.github/workflows/pages.yml` builds the site and deploys `site/dist` using official GitHub Pages Actions on pushes to `main` and manual workflow runs. PRs build but do not publish. Base path is `/PhiStream/`. Published URL, when GitHub Pages is enabled: https://michaelwave369.github.io/PhiStream/

If GitHub Pages isn't enabled, repository Settings → Pages → **Build and deployment** → **GitHub Actions**. Repository admins must configure this when necessary; the deployment workflow cannot promise to enable Pages automatically.

The **capacity explorer is arithmetic only** (parameter count × storage bytes/parameter) and reserves an illustrative slice of device memory. It is not a GPU feasibility calculator. All performance receipts are explicitly synthetic, CPU-only tests. No model files shipped.

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
