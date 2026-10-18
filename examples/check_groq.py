"""Context: diagnose Groq model access without uploading any financial disclosure.

Lists models and optionally sends one tiny JSON-only connectivity probe. Credentials
and provider error messages are never printed; only status and error codes are shown.
"""

import argparse
import json
from pathlib import Path

from groq import APIError, Groq

from financial_rag.generation import load_environment, resolve_model


def main():
    """Context: separate model discovery from a minimal, explicitly requested probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    load_environment()
    client = Groq(timeout=20, max_retries=0)
    model = resolve_model(args.model)
    report = {"configured_model": model}
    try:
        ids = sorted(item.id for item in client.models.list().data if getattr(item, "active", True))
        report.update(available_models=ids, configured_model_listed=model in ids)
        print(json.dumps(report, indent=2))
        if args.probe:
            options = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss-") else {}
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": 'Return JSON only: {"ok": true}'}],
                response_format={"type": "json_object"}, max_completion_tokens=300,
                temperature=0, **options)
            report["probe"] = {"finish_reason": response.choices[0].finish_reason,
                               "content": response.choices[0].message.content}
            print("Probe:", json.dumps(report["probe"]))
    except APIError as exc:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        report["error"] = {"status": getattr(exc, "status_code", None),
                           "code": error.get("code") if isinstance(error, dict) else None,
                           "type": type(exc).__name__}
        print("Diagnostic error:", json.dumps(report["error"]))
    Path("runs").mkdir(exist_ok=True)
    Path("runs/groq-model-access.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if "error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
