"""goal_only baseline on an arbitrary closed target (gpt-4o-mini / gpt-5-nano).

Each JMIR crisis utterance is sent to the target under the counseling system prompt
(no persona). The response is scored by the PCSA Figure-4 four-axis judge (gpt-4o-mini)
and classified CARES Refuse/Cautious/Accept. Reports strict 4-axis UNSAFE and CARES
ASR, stratified by crisis label. Parallel + resume-safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

from experiments.persona_pilot import atomic_json, SYSTEM
from experiments.model_io import respond
from experiments.pcsa_embedded_multiturn import judge_once
from experiments.pcsa_attack_optimize import cares_classify, CARES_SS
from evaluation.pcsa_judge import FIELDS, AXES
from select_by_surrogate import digest

import threading
_wlock = threading.Lock()


def load_goals(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def run_one(goal, target_model, base, max_out):
    utt = goal['goal']
    convo = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': utt}]
    resp = respond(target_model, convo, base=base, max_out=max_out)
    case = {'goal': goal, 'pathology': {'compliance_lever': ''}}
    j = judge_once(case, utt, resp['text'])
    cares = cares_classify(utt, '', resp['text'])
    return {'goal_id': goal['goal_id'], 'crisis_label': goal.get('crisis_label'),
            'axes': {AXES[k]: j[k] for k in FIELDS}, 'unsafe': any(j[k] for k in FIELDS),
            'cares_label': cares, 'cares_engage': int(cares != 'Refuse'), 'cares_accept': int(cares == 'Accept'),
            'cares_ss': CARES_SS[cares], 'finish': resp['finish_reason'],
            'resp_sha256': hashlib.sha256(resp['text'].encode()).hexdigest()}


def summarize(rows):
    n = len(rows)
    def blk(sub):
        m = len(sub)
        return {'n': m, 'unsafe': sum(r['unsafe'] for r in sub),
                'cares_engage': sum(r['cares_engage'] for r in sub),
                'cares_accept': sum(r['cares_accept'] for r in sub)} if m else {'n': 0}
    return {'overall': blk(rows),
            'axis_rates': {AXES[k]: round(sum(r['axes'][AXES[k]] for r in rows) / n, 4) for k in FIELDS} if n else {},
            'cares_label_counts': {l: sum(1 for r in rows if r['cares_label'] == l) for l in ('Refuse', 'Cautious', 'Accept')},
            'by_label': {lab: blk([r for r in rows if r['crisis_label'] == lab]) for lab in sorted({r['crisis_label'] for r in rows})}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target-model', required=True)
    ap.add_argument('--target-base-url', default='https://api.openai.com/v1')
    ap.add_argument('--goals', type=Path, default=ROOT / 'goals/crisis_goals_jmir_client.jsonl')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--max-out', type=int, default=900)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    goals = load_goals(args.goals)
    if args.limit:
        goals = goals[:args.limit]
    settings = {'mode': 'goal_only', 'target_model': args.target_model, 'n': len(goals),
                'judge': 'gpt-4o-mini-2024-07-18', 'goal_ids': [g['goal_id'] for g in goals],
                'src': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fp = digest(settings)
    d = args.out_dir; d.mkdir(parents=True, exist_ok=True)
    man = d / 'manifest.json'
    if args.resume and man.exists():
        if json.loads(man.read_text())['fingerprint'] != fp:
            raise ValueError('settings changed; new dir')
    elif man.exists():
        raise ValueError('run exists; --resume or new dir')
    else:
        atomic_json(man, {'fingerprint': fp, 'settings': settings})
    rp = d / 'results.jsonl'
    done = {json.loads(l)['goal_id'] for l in rp.read_text().splitlines()} if (args.resume and rp.exists()) else set()
    todo = [g for g in goals if g['goal_id'] not in done]
    out = rp.open('a', encoding='utf-8')
    skipped = []
    try:
        with ThreadPoolExecutor(max(1, args.workers)) as pool:
            futs = {pool.submit(run_one, g, args.target_model, args.target_base_url, args.max_out): g for g in todo}
            for i, f in enumerate(as_completed(futs), 1):
                g = futs[f]
                try:
                    row = f.result()
                except Exception as exc:
                    skipped.append({'goal_id': g['goal_id'], 'error': type(exc).__name__ + ': ' + str(exc)}); continue
                with _wlock:
                    out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
                if i % 25 == 0:
                    print(f'[{i}/{len(todo)}] skipped={len(skipped)}', flush=True)
    finally:
        out.close()
    if skipped:
        atomic_json(d / 'skipped.json', skipped)
    rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
    s = summarize(rows)
    summary = {'mode': 'goal_only', 'target_model': args.target_model, 'n_scored': len(rows),
               'n_skipped': len(skipped), **s,
               'note': 'goal_only (no persona). 4-axis = strict PCSA Fig-4; CARES = Refuse/Cautious/Accept.'}
    atomic_json(d / 'summary.json', summary)
    o = s['overall']
    print('=== GOAL_ONLY %s DONE ===' % args.target_model, flush=True)
    print(json.dumps({'n': o['n'], 'unsafe': o['unsafe'], 'cares_engage': o['cares_engage'],
                      'cares_accept': o['cares_accept'], 'skipped': len(skipped), 'axis_rates': s['axis_rates']}, ensure_ascii=False), flush=True)
    print('GOAL_ONLY %s FINALIZE COMPLETE' % args.target_model, flush=True)


if __name__ == '__main__':
    main()
