#!/usr/bin/env python3
"""Quickly verify Gemini model inference and fine-tuning on Vertex AI.

The script focuses on Google-published Gemini models that are relevant to
adapter-based supervised fine-tuning (SFT). For each model it can:

* confirm the documented support status (inference vs. fine-tuning)
* execute a lightweight inference request through the Vertex AI SDK
* optionally submit an SFT job (guarded by a CLI flag)
* capture detailed error diagnostics when requests fail

Optionally, AI Studio (google-generativeai) inference checks can be run for
models that have a public API alias.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    def load_dotenv(path: Optional[str] = None) -> None:
        """Minimal .env loader when python-dotenv is unavailable."""

        candidate = Path(path) if path else Path(".env")
        if not candidate.exists():
            return

        for raw_line in candidate.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass
class CapabilityResult:
    status: str
    detail: str
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelCapabilityReport:
    provider: str
    model_id: str
    display_name: Optional[str] = None
    ai_studio_model: Optional[str] = None
    inference_support: Optional[CapabilityResult] = None
    inference_vertex: Optional[CapabilityResult] = None
    inference_aistudio: Optional[CapabilityResult] = None
    fine_tune_support: Optional[CapabilityResult] = None
    fine_tune_result: Optional[CapabilityResult] = None


@dataclass
class InventoryReport:
    generated_at_utc: str
    project_id: Optional[str]
    region: Optional[str]
    vertex_models: List[ModelCapabilityReport]


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "gemini-1.5-pro-002": {
        "display_name": "Gemini 1.5 Pro (Adapter)",
        "inference_expected": True,
        "inference_note": "GA inference on Vertex AI.",
        "fine_tune_expected": True,
        "fine_tune_note": "Adapter-based SFT (preview) in us-central1. Allow-list required.",
        "ai_studio_model": "models/gemini-1.5-pro",
    },
    "gemini-1.5-flash-002": {
        "display_name": "Gemini 1.5 Flash (Adapter)",
        "inference_expected": True,
        "inference_note": "GA inference on Vertex AI.",
        "fine_tune_expected": True,
        "fine_tune_note": "Adapter-based SFT (preview) in us-central1. Allow-list required.",
        "ai_studio_model": "models/gemini-1.5-flash",
    },
    "gemini-1.0-pro-002": {
        "display_name": "Gemini 1.0 Pro",
        "inference_expected": True,
        "inference_note": "Legacy inference. Fine-tuning no longer available.",
        "fine_tune_expected": False,
        "fine_tune_note": "SFT retired as of Feb 2025; expect INVALID_ARGUMENT.",
        "ai_studio_model": "models/gemini-pro",
    },
    "gemini-2.0-flash": {
        "display_name": "Gemini 2.0 Flash",
        "inference_expected": True,
        "inference_note": "GA inference only; no adapter tuning.",
        "fine_tune_expected": False,
        "fine_tune_note": "Fine-tuning unavailable for Gemini 2.x as of Feb 2025.",
        "ai_studio_model": "models/gemini-2.0-flash-exp",
    },
    "text-bison@002": {
        "display_name": "Text Bison 002 (Legacy)",
        "inference_expected": True,
        "inference_note": "Legacy PaLM-based text model.",
        "fine_tune_expected": True,
        "fine_tune_note": "Classic SFT path; slower but still supported.",
        "ai_studio_model": None,
    },
}

DEFAULT_MODEL_ORDER: Sequence[str] = tuple(MODEL_SPECS.keys())


def _safe_import_google_modules():
    try:
        import google.generativeai as genai  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "google-generativeai package is required. Install via `pip install google-generativeai`."
        ) from exc

    try:
        import vertexai  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore
        from vertexai.preview.tuning import sft  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "vertexai package is required. Install via `pip install google-cloud-aiplatform`."
        ) from exc

    try:
        from google.api_core import exceptions as google_exceptions  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-api-core package is missing. Install via `pip install google-api-core`."
        ) from exc

    return {
        "genai": genai,
        "vertexai": vertexai,
        "GenerativeModel": GenerativeModel,
        "sft": sft,
        "google_exceptions": google_exceptions,
    }


def _load_env_file(env_path: Optional[str]) -> None:
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()


def _ensure_env_defaults() -> None:
    os.environ.setdefault("GOOGLE_CLOUD_DISABLE_GRPC_ALTS", "1")


def _validate_env(project: Optional[str], region: Optional[str], credentials_path: Optional[str]) -> List[str]:
    warnings: List[str] = []

    if not project:
        warnings.append("GOOGLE_CLOUD_PROJECT not set -> Vertex AI requests will fail.")
    if not region:
        warnings.append("GOOGLE_CLOUD_REGION not set; defaulting to us-central1.")
    elif region != "us-central1":
        warnings.append("Gemini adapter tuning only deploys in us-central1; other regions will fail.")

    if credentials_path:
        credentials = Path(credentials_path)
        if not credentials.exists():
            warnings.append(f"Credentials file not found: {credentials}")
    else:
        warnings.append("GOOGLE_APPLICATION_CREDENTIALS not set -> SDK will use default gcloud account.")

    return warnings


def _truncate(text: str, limit: int = 280) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarise_exception(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _diagnose_message(message: str) -> str:
    lowered = message.lower()
    if "permission" in lowered or "not authorized" in lowered:
        return "Check Vertex AI API enablement, billing status, and IAM roles/service account permissions."
    if "not supported" in lowered:
        return "Model does not support this operation in the selected project/region."
    if "not found" in lowered:
        return "Model identifier unavailable in this region or project; confirm spelling and allow-list."
    if "resource exhausted" in lowered or "quota" in lowered:
        return "Quota or rate limit hit; review Vertex AI quotas in Cloud Console."
    if "location" in lowered and "invalid" in lowered:
        return "Operation requested in unsupported region; use us-central1 for Gemini SFT."
    if "unsupported tuning_spec" in lowered:
        return "Attempted fine-tuning path is not enabled; adapter SFT only works for 1.5 adapter SKUs."
    return "See detail for more information."


def _build_reports(model_ids: Sequence[str]) -> List[ModelCapabilityReport]:
    reports: List[ModelCapabilityReport] = []
    for model_id in model_ids:
        spec = MODEL_SPECS.get(model_id, {})
        report = ModelCapabilityReport(
            provider="vertex-ai",
            model_id=model_id,
            display_name=spec.get("display_name"),
            ai_studio_model=spec.get("ai_studio_model"),
        )

        inference_expected = spec.get("inference_expected")
        if inference_expected is None:
            report.inference_support = CapabilityResult(
                status="unknown",
                detail="No catalog information."
            )
        else:
            report.inference_support = CapabilityResult(
                status="supported" if inference_expected else "unsupported",
                detail=str(spec.get("inference_note", "")),
            )

        fine_tune_expected = spec.get("fine_tune_expected")
        if fine_tune_expected is None:
            report.fine_tune_support = CapabilityResult(
                status="unknown",
                detail="Fine-tuning status not documented."
            )
        else:
            report.fine_tune_support = CapabilityResult(
                status="supported" if fine_tune_expected else "unsupported",
                detail=str(spec.get("fine_tune_note", "")),
            )

        reports.append(report)

    return reports


def run_vertex_inference_tests(
    sdk_modules: Dict[str, object],
    reports: Iterable[ModelCapabilityReport],
    prompt: str,
) -> None:
    GenerativeModel = sdk_modules["GenerativeModel"]
    google_exceptions = sdk_modules["google_exceptions"]

    for report in reports:
        try:
            model = GenerativeModel(report.model_id)
            response = model.generate_content(prompt)
            text = getattr(response, "text", "") or str(response)
            report.inference_vertex = CapabilityResult(
                status="success",
                detail=_truncate(text or "<empty response>"),
            )
        except google_exceptions.GoogleAPIError as exc:
            diagnosis = _diagnose_message(str(exc))
            report.inference_vertex = CapabilityResult(
                status="api_error",
                detail=_summarise_exception(exc),
                extra={"diagnosis": diagnosis},
            )
        except Exception as exc:  # pylint: disable=broad-except
            report.inference_vertex = CapabilityResult(
                status="error",
                detail=_summarise_exception(exc),
                extra={"diagnosis": "Unhandled client-side exception."},
            )


def run_ai_studio_inference_tests(
    genai_module,
    reports: Iterable[ModelCapabilityReport],
    prompt: str,
) -> None:
    for report in reports:
        if not report.ai_studio_model:
            continue
        try:
            model = genai_module.GenerativeModel(report.ai_studio_model)
            response = model.generate_content(prompt)
            text = getattr(response, "text", "") or str(response)
            report.inference_aistudio = CapabilityResult(
                status="success",
                detail=_truncate(text or "<empty response>"),
            )
        except Exception as exc:  # pylint: disable=broad-except
            report.inference_aistudio = CapabilityResult(
                status="error",
                detail=_summarise_exception(exc),
                extra={"diagnosis": _diagnose_message(str(exc))},
            )


def attempt_vertex_fine_tune(
    sdk_modules: Dict[str, object],
    reports: Iterable[ModelCapabilityReport],
    training_dataset: str,
    tuned_model_display_name: Optional[str],
    fine_tune_targets: Sequence[str],
) -> None:
    sft = sdk_modules["sft"]
    google_exceptions = sdk_modules["google_exceptions"]
    target_set = set(fine_tune_targets)

    for report in reports:
        if report.model_id not in target_set:
            continue

        display_name = tuned_model_display_name or (
            f"probe-{report.model_id}-{datetime.utcnow():%Y%m%d%H%M%S}"
        )

        try:
            job = sft.train(
                source_model=report.model_id,
                train_dataset=training_dataset,
                tuned_model_display_name=display_name,
            )
            report.fine_tune_result = CapabilityResult(
                status="job_submitted",
                detail=f"Tuning job started: {job.resource_name}",
                extra={
                    "tuned_model_name": job.tuned_model_name or "<pending>",
                    "tuned_model_endpoint": job.tuned_model_endpoint_name or "<pending>",
                },
            )
        except google_exceptions.GoogleAPIError as exc:
            report.fine_tune_result = CapabilityResult(
                status="api_error",
                detail=_summarise_exception(exc),
                extra={"diagnosis": _diagnose_message(str(exc))},
            )
        except Exception as exc:  # pylint: disable=broad-except
            report.fine_tune_result = CapabilityResult(
                status="error",
                detail=_summarise_exception(exc),
                extra={"diagnosis": "Unhandled client-side exception."},
            )


def _report_to_dict(report: InventoryReport) -> Dict[str, object]:
    def _maybe_asdict(cap: Optional[CapabilityResult]):
        return asdict(cap) if cap else None

    def _model_asdict(model: ModelCapabilityReport):
        payload = {
            "provider": model.provider,
            "model_id": model.model_id,
            "display_name": model.display_name,
            "ai_studio_model": model.ai_studio_model,
            "inference_support": _maybe_asdict(model.inference_support),
            "inference_vertex": _maybe_asdict(model.inference_vertex),
            "inference_aistudio": _maybe_asdict(model.inference_aistudio),
            "fine_tune_support": _maybe_asdict(model.fine_tune_support),
            "fine_tune_result": _maybe_asdict(model.fine_tune_result),
        }
        return payload

    return {
        "generated_at_utc": report.generated_at_utc,
        "project_id": report.project_id,
        "region": report.region,
        "vertex_models": [_model_asdict(m) for m in report.vertex_models],
    }


def _summaries(report: InventoryReport) -> Dict[str, Dict[str, int]]:
    def summarise(collection: Iterable[ModelCapabilityReport], accessor: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for model in collection:
            result: Optional[CapabilityResult] = getattr(model, accessor)
            key = result.status if result else "not_run"
            counts[key] = counts.get(key, 0) + 1
        return counts

    return {
        "vertex_inference": summarise(report.vertex_models, "inference_vertex"),
        "vertex_fine_tune": summarise(report.vertex_models, "fine_tune_result"),
        "ai_studio_inference": summarise(report.vertex_models, "inference_aistudio"),
    }


def _print_human_summary(report: InventoryReport) -> None:
    summaries = _summaries(report)

    print("\nVertex inference status counts:", summaries["vertex_inference"])
    print("Vertex fine-tuning status counts:", summaries["vertex_fine_tune"])
    if any(model.inference_aistudio for model in report.vertex_models):
        print("AI Studio inference status counts:", summaries["ai_studio_inference"])

    print("\nDetailed results:")
    for model in report.vertex_models:
        name = model.display_name or model.model_id
        print(f"- {name} ({model.model_id})")
        if model.inference_support:
            print(
                f"    Documented inference: {model.inference_support.status} - {model.inference_support.detail}"
            )
        if model.fine_tune_support:
            print(
                f"    Documented fine-tune: {model.fine_tune_support.status} - {model.fine_tune_support.detail}"
            )
        if model.inference_vertex:
            print(f"    Vertex inference: {model.inference_vertex.status} - {model.inference_vertex.detail}")
            if model.inference_vertex.extra.get("diagnosis"):
                print(f"      Diagnosis: {model.inference_vertex.extra['diagnosis']}")
        if model.fine_tune_result:
            print(f"    Fine-tune: {model.fine_tune_result.status} - {model.fine_tune_result.detail}")
            if model.fine_tune_result.extra.get("diagnosis"):
                print(f"      Diagnosis: {model.fine_tune_result.extra['diagnosis']}")
        if model.inference_aistudio:
            print(f"    AI Studio: {model.inference_aistudio.status} - {model.inference_aistudio.detail}")
            if model.inference_aistudio.extra.get("diagnosis"):
                print(f"      Diagnosis: {model.inference_aistudio.extra['diagnosis']}")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Path to .env containing Google credentials.")
    parser.add_argument(
        "--models",
        action="append",
        help="Model IDs to probe (defaults to curated Gemini set). Repeat flag for multiple models.",
    )
    parser.add_argument(
        "--prompt",
        default="What is a large language model?",
        help="Prompt used for inference smoke tests.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip live inference tests (still prints documented support).",
    )
    parser.add_argument(
        "--include-ai-studio",
        action="store_true",
        help="Run AI Studio inference for models with a public API alias (requires GOOGLE_API_KEY).",
    )
    parser.add_argument(
        "--submit-fine-tune",
        action="store_true",
        help="Submit Vertex AI SFT jobs for supported models (uses --fine-tune-model selection).",
    )
    parser.add_argument(
        "--fine-tune-model",
        action="append",
        help="Model IDs to fine-tune when --submit-fine-tune is set. Defaults to adapter-capable models.",
    )
    parser.add_argument(
        "--training-dataset",
        default="gs://cloud-samples-data/vertex-ai/model-evaluation/peft_train_sample.jsonl",
        help="GCS URI of JSONL dataset for tuning jobs.",
    )
    parser.add_argument(
        "--tuned-model-display-name",
        help="Optional display name override for tuning jobs.",
    )
    parser.add_argument(
        "--report-file",
        help="Write JSON report to this path (directories created as needed).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    _load_env_file(args.env_file)
    _ensure_env_defaults()

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    api_key = os.environ.get("GOOGLE_API_KEY")

    warnings = _validate_env(project_id, region, credentials_path)
    for warning in warnings:
        print(f"[WARN] {warning}", file=sys.stderr)

    modules = _safe_import_google_modules()

    modules["vertexai"].init(project=project_id, location=region)

    if args.include_ai_studio:
        if not api_key:
            print("[WARN] GOOGLE_API_KEY not set; skipping AI Studio checks.", file=sys.stderr)
            args.include_ai_studio = False
        else:
            modules["genai"].configure(api_key=api_key)

    model_list = args.models or list(DEFAULT_MODEL_ORDER)
    reports = _build_reports(model_list)

    if not args.skip_inference:
        run_vertex_inference_tests(modules, reports, args.prompt)
    else:
        for report in reports:
            report.inference_vertex = CapabilityResult(
                status="skipped",
                detail="Inference skipped via CLI flag.",
            )

    if args.include_ai_studio:
        run_ai_studio_inference_tests(modules["genai"], reports, args.prompt)

    if args.submit_fine_tune:
        fine_tune_targets = args.fine_tune_model or [
            model_id
            for model_id, spec in MODEL_SPECS.items()
            if spec.get("fine_tune_expected")
        ]
        attempt_vertex_fine_tune(
            modules,
            reports,
            training_dataset=args.training_dataset,
            tuned_model_display_name=args.tuned_model_display_name,
            fine_tune_targets=fine_tune_targets,
        )
    else:
        for report in reports:
            if report.fine_tune_support and report.fine_tune_support.status == "supported":
                report.fine_tune_result = CapabilityResult(
                    status="skipped",
                    detail="Fine-tune skipped. Use --submit-fine-tune to launch a job.",
                )

    report = InventoryReport(
        generated_at_utc=datetime.utcnow().isoformat() + "Z",
        project_id=project_id,
        region=region,
        vertex_models=reports,
    )

    payload = _report_to_dict(report)
    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2))
        print(f"Report written to {report_path}")
    else:
        print(json.dumps(payload, indent=2))

    _print_human_summary(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
