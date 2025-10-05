# Model Inventory Probe

This utility exercises a curated list of Google Gemini models using the
credentials in your `.env`. It verifies documented support, runs optional
Vertex AI inference smoke tests, and can submit supervised fine-tuning jobs
to confirm that your project is entitled to adapter-based SFT.

## Requirements

- Python 3.9+
- Packages: `google-cloud-aiplatform`, `google-generativeai`, `python-dotenv`
- Environment variables (see repository `.env`):
  - `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION` (default: `us-central1`)
  - `GOOGLE_APPLICATION_CREDENTIALS`
  - `GOOGLE_API_KEY` (required for AI Studio probing)

## Usage

From the repository root:

```bash
python tests/model_probing/model_inventory_report.py \
  --env-file .env \
  --report-file reports/model_probe.json
```

Optional flags:

- `--models MODEL_ID` (repeatable) to override the default Gemini set.
- `--include-ai-studio` to run the checks against the AI Studio API as well.
- `--submit-fine-tune` to launch adapter SFT jobs (defaults to adapter SKUs).
- `--fine-tune-model MODEL_ID` (repeatable) to target specific models when
  `--submit-fine-tune` is used.
- `--training-dataset` to override the default public JSONL dataset.
- `--tuned-model-display-name` to set a custom display name for jobs.
- `--skip-inference` to only print the documented support table.

## Output

- JSON report (stdout or `--report-file`).
- Console summary of inference/tuning status counts.

`MODEL_ID` values correspond to publisher model IDs (for example,
`gemini-1.5-pro-002`). Use the JSON report to confirm which models
support supervised tuning before launching a job.
