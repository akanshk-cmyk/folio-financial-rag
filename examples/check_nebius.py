"""Context: verify Nebius authentication and JSON completion without exposing keys."""

import json
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from financial_rag.generation import load_environment, resolve_model, nebius_base_url, request_nebius


def main():
    """Context: make one small generation request before starting a paid evaluation batch."""
    load_environment()
    report = {"provider": "nebius", "model": resolve_model(provider="nebius"),
              "judge_model": resolve_model(judge=True, provider="nebius"),
              "base_url": nebius_base_url(), "key_configured": bool(os.getenv("NEBIUS_API_KEY"))}
    try:
        if not report["key_configured"]:
            raise ValueError("Set NEBIUS_API_KEY locally in .env.")
        prompt = ChatPromptValue(messages=[SystemMessage(content="Return only a JSON object."),
            HumanMessage(content='Return exactly {"ok":true}.')])
        response = request_nebius(prompt, report["model"], max_completion_tokens=512)
        report["completion_access"] = bool(response.choices and
            response.choices[0].finish_reason == "stop" and
            json.loads(response.choices[0].message.content).get("ok") is True)
        report["usage"] = response.usage.model_dump() if response.usage else {}
    except (RuntimeError, ValueError) as exc:
        # Context: only our sanitized transport error is recorded; no provider bodies or credentials.
        report["error"] = str(exc) if isinstance(exc, RuntimeError) else "Missing key or invalid JSON response."
    path = Path("runs/apple-pinecone-benchmark-v1/nebius-access.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report.get("completion_access") else 1


if __name__ == "__main__":
    raise SystemExit(main())
