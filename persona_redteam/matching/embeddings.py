"""Standalone text embeddings for candidate retrieval; no strategy optimizer."""
import json
import os
import urllib.request


def embed_texts(texts, model="text-embedding-3-small", batch=256):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    vectors = []
    for start in range(0, len(texts), batch):
        request = urllib.request.Request("https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": model, "input": texts[start:start + batch]}).encode(),
            method="POST", headers={"Authorization": "Bearer " + key,
                                    "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = sorted(json.load(response)["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in data)
    return vectors
