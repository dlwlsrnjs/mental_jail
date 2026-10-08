"""Model-aware chat completion for closed targets, handling reasoning models.

gpt-5* / o1* / o3* reject `max_tokens` and non-default `temperature`, and spend
tokens on hidden reasoning, so they need `max_completion_tokens` with a generous
budget. gpt-4o* use `max_tokens` + temperature. One helper so target responders can
be swapped (gpt-4o-mini / gpt-5-nano / local surrogate) without per-call branching.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

OPENAI_BASE = 'https://api.openai.com/v1'


def _primary_key():
    from experiments.persona_pilot import primary_key
    return primary_key()


def is_reasoning(model: str) -> bool:
    return model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3') or model.startswith('o4')


def respond(model, messages, base=OPENAI_BASE, max_out=900, reasoning_budget=6000):
    """Return {'text','model','finish_reason','usage'} or raise. Picks parameters by model."""
    local = urlparse(base).hostname in ('localhost', '127.0.0.1', '::1')
    body = {'model': model, 'messages': messages}
    if is_reasoning(model):
        body['max_completion_tokens'] = max(max_out, reasoning_budget)
    else:
        body['max_tokens'] = max_out
        body['temperature'] = 0
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions',
        data=json.dumps(body, ensure_ascii=False).encode(), method='POST',
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + ('local' if local else _primary_key())})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            v = json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError('generation HTTP ' + str(exc.code) + ': ' + exc.read().decode()[:200]) from None
    choice = v['choices'][0]
    text = choice['message'].get('content')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('empty completion or refused (finish=%s)' % choice.get('finish_reason'))
    return {'text': text, 'model': v.get('model'), 'finish_reason': choice.get('finish_reason'),
            'usage': v.get('usage', {})}
