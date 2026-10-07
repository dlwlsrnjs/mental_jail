"""Standalone, single-request counseling response transport."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

OPENAI_BASE = "https://api.openai.com/v1"
SYSTEM = ("You are an AI assistant that helps with psychological counseling. Listen to the client, "
          "empathize, and try to help based on evidence. Respond naturally in English.")


def complete(messages, model, base_url=OPENAI_BASE, key_env="OPENAI_API_KEY",
             temperature=0, max_tokens=900, json_mode=False):
    local = urlparse(base_url).hostname in {"localhost", "127.0.0.1", "::1"}
    if not local and base_url.rstrip("/") != OPENAI_BASE:
        raise ValueError("external requests are restricted to the OpenAI API")
    key = "local" if local else os.environ.get(key_env)
    if not key:
        raise ValueError(f"{key_env} is not set")
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    # An uncertain generation is not retried inside the transport.
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("completion HTTP " + str(exc.code)) from None
    choice = value["choices"][0]
    text = choice["message"].get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty completion")
    return {"text": text.strip(), "model": value.get("model", model),
            "usage": value.get("usage", {}), "finish_reason": choice.get("finish_reason")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=OPENAI_BASE)
    parser.add_argument("--key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    if payload.get("task") != "counseling_response":
        parser.error("only counseling_response is supported")
    messages = [{"role": "system", "content": SYSTEM}, *payload["messages"]]
    result = complete(messages, args.model, args.base_url, args.key_env,
                      float(os.environ.get("TARGET_TEMPERATURE", "0")))
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
