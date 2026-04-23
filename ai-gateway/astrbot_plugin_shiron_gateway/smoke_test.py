from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shiron smoke test script")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:3000",
        help="Shiron base url, e.g. http://127.0.0.1:3000 or https://host/v1",
    )
    parser.add_argument("--api-key", required=True, help="Shiron API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model name")
    parser.add_argument("--prompt", default="hello", help="Prompt for chat/responses test")
    parser.add_argument(
        "--tests",
        default="models,chat,responses",
        help="Comma-separated tests: models,chat,responses,completions",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout")
    return parser.parse_args()


def normalize_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    norm = path if path.startswith("/") else f"/{path}"
    if base.endswith("/v1") and norm.startswith("/v1/"):
        norm = norm[3:]
    return f"{base}{norm}"


def extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message", {})
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def extract_responses_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output = payload.get("output", [])
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for seg in content:
            if isinstance(seg, dict):
                text = seg.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    tests = {x.strip().lower() for x in args.tests.split(",") if x.strip()}
    headers = {"Authorization": f"Bearer {args.api_key}"}

    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        if "models" in tests:
            print("== models ==")
            resp = client.get(normalize_url(args.base_url, "/v1/models"))
            print(f"HTTP {resp.status_code}")
            payload = resp.json()
            models = payload.get("data", []) if isinstance(payload, dict) else []
            print(f"model_count={len(models)}")
            if models:
                first = models[0]
                if isinstance(first, dict):
                    print(f"first_model={first.get('id')}")
            if resp.status_code >= 400:
                print_json(payload)
                return 1

        if "chat" in tests:
            print("== chat/completions ==")
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": args.prompt}],
                "stream": False,
            }
            resp = client.post(normalize_url(args.base_url, "/v1/chat/completions"), json=body)
            print(f"HTTP {resp.status_code}")
            payload = resp.json()
            if resp.status_code >= 400:
                print_json(payload)
                return 1
            print(extract_chat_text(payload) or "[empty chat text]")

        if "responses" in tests:
            print("== responses ==")
            body = {"model": args.model, "input": args.prompt}
            resp = client.post(normalize_url(args.base_url, "/v1/responses"), json=body)
            print(f"HTTP {resp.status_code}")
            payload = resp.json()
            if resp.status_code >= 400:
                print_json(payload)
                return 1
            print(extract_responses_text(payload) or "[empty responses text]")

        if "completions" in tests:
            print("== completions ==")
            body = {"model": args.model, "prompt": args.prompt}
            resp = client.post(normalize_url(args.base_url, "/v1/completions"), json=body)
            print(f"HTTP {resp.status_code}")
            payload = resp.json()
            if resp.status_code >= 400:
                print_json(payload)
                return 1
            choices = payload.get("choices", []) if isinstance(payload, dict) else []
            if choices and isinstance(choices[0], dict):
                print(choices[0].get("text", "[empty completion text]"))
            else:
                print("[empty completion text]")

    print("Smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
