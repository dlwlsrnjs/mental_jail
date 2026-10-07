"""Serve the pinned, downloaded Llama surrogate on loopback only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-record', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/llama31_surrogate_model.json')
    parser.add_argument('--port', type=int, default=8014)
    args = parser.parse_args()
    record = json.loads(args.model_record.read_text())
    if record['model'] != 'meta-llama/Llama-3.1-8B-Instruct':
        parser.error('this server is restricted to the requested official surrogate')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = record['snapshot_path']
    if Path(path).name != record['revision']:
        parser.error('snapshot/revision mismatch')
    print('Loading pinned surrogate', record['model'], record['revision'], flush=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                               torch_dtype=torch.bfloat16).eval().to('cuda')
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value):
            body = json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path.rstrip('/') == '/v1/models':
                self.send(200, {'data': [{'id': record['model'], 'revision': record['revision']}],
                                'weight_snapshot': record['revision']})
            else:
                self.send(404, {'error': 'not found'})

        def do_POST(self):
            if self.path.rstrip('/') != '/v1/chat/completions':
                return self.send(404, {'error': 'not found'})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                if body.get('model') != record['model']:
                    return self.send(400, {'error': 'model identity mismatch'})
                max_tokens = int(body.get('max_tokens', 900))
                if not 1 <= max_tokens <= 2048:
                    return self.send(400, {'error': 'invalid output limit'})
                temperature = float(body.get('temperature', 0))
                with lock, torch.inference_mode():
                    enc = tok.apply_chat_template(body['messages'], add_generation_prompt=True,
                                                   return_tensors='pt', return_dict=True)
                    enc = {k: v.to(model.device) for k, v in enc.items()}
                    start = enc['input_ids'].shape[-1]
                    options = {'max_new_tokens': max_tokens, 'pad_token_id': tok.eos_token_id,
                               'do_sample': temperature > 0}
                    if temperature > 0:
                        options['temperature'] = temperature
                    output = model.generate(**enc, **options)
                    tail = output[0][start:]
                    text = tok.decode(tail, skip_special_tokens=True).strip()
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else eos or [tok.eos_token_id]
                    finish = 'stop' if int(tail[-1]) in eos else 'length'
                    completion_tokens = len(tail)
                self.send(200, {'id': 'local-' + str(time.time_ns()), 'model': record['model'],
                                'revision': record['revision'],
                                'choices': [{'message': {'role': 'assistant', 'content': text},
                                             'finish_reason': finish, 'index': 0}],
                                'usage': {'prompt_tokens': int(start),
                                          'completion_tokens': completion_tokens,
                                          'total_tokens': int(start) + completion_tokens}})
            except Exception as exc:
                self.send(500, {'error': type(exc).__name__})

    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('Pinned surrogate ready on port', args.port, flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
