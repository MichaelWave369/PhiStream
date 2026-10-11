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
