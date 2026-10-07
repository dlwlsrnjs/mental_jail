"""Offline checks for selection validity and response-preserving resume."""
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

import compliance_judge as judge
import select_by_surrogate as selector


def evaluation(score=0, evidence=None):
    return {"score": score, "unsafe_compliance": float(score) > 0,
            "unsafe_validation": False, "refusal": False,
            "evidence": evidence or [], "reason": "fixture"}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "trials.jsonl"
        self.case = selector.load_inputs(
            selector.ROOT / "goals/crisis_goals_jmir_client.jsonl",
            selector.ROOT / "outputs/goal_pathology_persona_routed_n813.jsonl",
            selector.ROOT / "personas", limit=1)[0]

    def log(self, resume=False):
        log = selector.TrialLog(self.path, "fixture-fingerprint", resume)
        self.addCleanup(log.close)
        return log

    def test_highest_response_score_wins_even_with_lower_cosine(self):
        expected = self.case["candidates"][-1]["candidate"]["id"]
        prompts = {selector.wrap_goal(self.case, x): x["candidate"]["id"] for x in self.case["candidates"]}
        def surrogate(prompt):
            return "fixture response " + prompts.get(prompt, "baseline")
        def score(payload):
            s = "fixture response " + expected
            return evaluation("8", [s]) if payload["response"] == s else evaluation()
        row, failures = selector.evaluate_case(self.case, self.log(), surrogate, score)
        self.assertFalse(failures)
        self.assertEqual(row["selected_persona"]["id"], expected)
        self.assertEqual(row["evaluation"]["score"], 8)
        self.assertEqual(row["delta_vs_goal_only"], 8)

    def test_safe_support_is_zero_and_ties_do_not_use_cosine(self):
        self.case["candidates"] = list(reversed(self.case["candidates"]))
        row, failures = selector.evaluate_case(
            self.case, self.log(), lambda _: "Safe supportive response.", lambda _: evaluation())
        self.assertFalse(failures)
        self.assertEqual(row["selection_status"], "no_unsafe_signal")
        ids = sorted(e["candidate"]["id"] for e in self.case["candidates"])
        self.assertEqual(row["selected_persona"]["id"], ids[0])
        self.assertEqual(row["tied_personas"], ids)
        self.assertEqual(row["delta_vs_goal_only"], 0)

    def test_judge_failure_resume_reuses_every_surrogate_response(self):
        calls = {"surrogate": 0, "judge": 0}
        def surrogate(_):
            calls["surrogate"] += 1
            return "Safe support."
        def broken(_):
            calls["judge"] += 1
            if calls["judge"] == 2:
                raise RuntimeError("temporary judge outage")
            return evaluation()
        log = self.log()
        row, failures = selector.evaluate_case(self.case, log, surrogate, broken)
        self.assertIsNone(row)
        self.assertEqual(len(failures), 1)
        self.assertEqual(calls["surrogate"], 4)
        log.close()
        def retry(_):
            calls["judge"] += 1
            return evaluation()
        resumed = self.log(resume=True)
        row, failures = selector.evaluate_case(self.case, resumed, surrogate, retry)
        self.assertFalse(failures)
        self.assertEqual(calls, {"surrogate": 4, "judge": 5})
        row2, failures = selector.evaluate_case(self.case, resumed, surrogate, retry)
        self.assertEqual(row, row2)
        self.assertEqual(calls, {"surrogate": 4, "judge": 5})

    def test_failed_candidate_cannot_be_scored_as_safe_or_omitted(self):
        prompt = selector.wrap_goal(self.case, self.case["candidates"][1])
        def surrogate(text):
            if text == prompt:
                raise TimeoutError("fixture")
            return "Safe support."
        row, failures = selector.evaluate_case(self.case, self.log(), surrogate, lambda _: evaluation())
        self.assertIsNone(row)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["stage"], "surrogate")

    def test_fingerprint_and_prompt_changes_are_rejected(self):
        log = self.log()
        selector.evaluate_case(self.case, log, lambda _: "Safe support.", lambda _: evaluation())
        log.close()
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            selector.TrialLog(self.path, "different", resume=True)
        changed = copy.deepcopy(self.case)
        changed["goal"]["goal"] += " changed"
        row, failures = selector.evaluate_case(changed, self.log(resume=True), lambda _: self.fail(), lambda _: self.fail())
        self.assertIsNone(row)
        self.assertEqual(len(failures), 4)

    def test_only_truncated_last_json_line_is_recoverable(self):
        log = self.log()
        row, _ = selector.evaluate_case(self.case, log, lambda _: "Safe support.", lambda _: evaluation())
        log.close()
        with self.path.open("a") as f:
            f.write('{"phase":')
        resumed = self.log(resume=True)
        row2, failures = selector.evaluate_case(self.case, resumed, lambda _: self.fail(), lambda _: self.fail())
        self.assertFalse(failures)
        self.assertEqual(row, row2)
        resumed.close()
        with self.path.open("a") as f:
            f.write("invalid line\n")
        with self.assertRaisesRegex(ValueError, "malformed trial log"):
            selector.TrialLog(self.path, "fixture-fingerprint", resume=True)

    def test_cleaned_goal_join_and_risk_validation(self):
        full = selector.load_inputs(selector.ROOT / "goals/crisis_goals_jmir_client.jsonl",
            selector.ROOT / "outputs/goal_pathology_persona_routed_n813.jsonl", selector.ROOT / "personas", scope="all")
        self.assertEqual(len(full), 652)
        path = Path(self.temporary.name) / "routed.jsonl"
        record = selector.read_jsonl(selector.ROOT / "outputs/goal_pathology_persona_routed_n813.jsonl")[0]
        record["goal"] += " changed"
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "text/label mismatch"):
            selector.load_inputs(selector.ROOT / "goals/crisis_goals_jmir_client.jsonl", path, selector.ROOT / "personas", limit=1)
        record["goal"] = self.case["goal"]["goal"]
        record["persona_candidates"][0]["id"] = "Noah"  # native Low risk, excluded for suicide route
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "risk filter"):
            selector.load_inputs(selector.ROOT / "goals/crisis_goals_jmir_client.jsonl", path, selector.ROOT / "personas", limit=1)

    def test_goal_text_preserved_and_failure_frame_not_sent_to_surrogate(self):
        case = copy.deepcopy(self.case)
        case["pathology"]["compliance_lever"] = "judge-only-marker"
        for entry in case["candidates"]:
            prompt = selector.wrap_goal(case, entry)
            self.assertTrue(prompt.endswith(case["goal"]["goal"]))
            self.assertNotIn("judge-only-marker", prompt)

    def test_dry_run_does_not_create_outputs_or_need_credentials(self):
        out = Path(self.temporary.name) / "selected.jsonl"
        result = subprocess.run([sys.executable, str(selector.ROOT / "matching/select_by_surrogate.py"),
            "--scope", "all", "--dry-run", "--out", str(out)], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["goals"], 652)
        self.assertEqual(plan["surrogate_calls"], 2608)
        self.assertEqual(plan["target_calls"], 0)
        self.assertFalse(out.exists())


class JudgeTests(unittest.TestCase):
    def test_safe_empathy_does_not_require_refusal(self):
        self.assertEqual(judge.validate_evaluation(evaluation())["score"], 0)

    def test_invalid_or_unsupported_judgments_are_not_positive_scores(self):
        cases = [evaluation(float("nan")), evaluation(11), evaluation(True), evaluation(5),
                 {**evaluation(), "unsafe_validation": True},
                 {**evaluation(5, ["quote"]), "refusal": True},
                 {**evaluation(), "unsafe_compliance": "false"}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                judge.validate_evaluation(case)
        with self.assertRaisesRegex(ValueError, "observed response"):
            judge.validate_response_evaluation(evaluation(5, ["invented quote"]), "Safe support.")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.out = root / "selected.jsonl"
        self.judge_path = root / "judge_fixture.py"
        self.judge_path.write_text("import json,sys\njson.load(sys.stdin)\nprint(" + repr(json.dumps(evaluation())) + ")\n")
        self.requests = []
        requests = self.requests
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def send(self, value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                self.send({"data": [{"id": "fixture-surrogate"}]})
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send({"choices": [{"message": {"content": "Safe support."}}]})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.command = [sys.executable, str(selector.ROOT / "matching/select_by_surrogate.py"),
            "--limit", "1", "--out", str(self.out), "--surrogate-model", "fixture-surrogate",
            "--surrogate-base-url", f"http://127.0.0.1:{self.server.server_port}/v1",
            "--surrogate-key-env", "FIXTURE_KEY", "--judge-command",
            json.dumps([sys.executable, str(self.judge_path)])]
        self.env = dict(os.environ, FIXTURE_KEY="local", TARGET_MODEL="must-never-be-used",
                        TARGET_BASE_URL="http://127.0.0.1:1/v1")

    def run_cli(self, extras=()):
        return subprocess.run([*self.command, *extras], env=self.env, capture_output=True, text=True, timeout=15)

    def test_cli_resume_never_repeats_completed_model_calls(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual({r['model'] for r in self.requests}, {'fixture-surrogate'})
        original = self.out.read_text()
        result = self.run_cli(["--resume"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(self.out.read_text(), original)

    def test_cli_refuses_changed_judge_source_on_resume(self):
        self.assertEqual(self.run_cli().returncode, 0)
        self.judge_path.write_text(self.judge_path.read_text() + "# changed\n")
        result = self.run_cli(["--resume"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("configuration/inputs changed", result.stderr)
        self.assertEqual(len(self.requests), 4)

    def test_cli_refuses_a_different_served_model_before_any_query(self):
        self.command[self.command.index("--surrogate-model") + 1] = "different-model"
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("surrogate identity check failed", result.stderr)
        self.assertFalse(self.requests)
        self.assertFalse(self.out.with_suffix('.manifest.json').exists())

    def test_rejudging_reuses_responses_but_not_old_judgments(self):
        self.assertEqual(self.run_cli().returncode, 0)
        cache = self.out.with_suffix('.trials.jsonl')
        new_out = self.out.with_name('rejudged.jsonl')
        self.command[self.command.index('--out') + 1] = str(new_out)
        # A new trusted judge still sees all four cached responses.
        counter = Path(self.temporary.name) / 'judge_calls.txt'
        self.judge_path.write_text(self.judge_path.read_text() +
            f"from pathlib import Path\nwith Path({str(counter)!r}).open('a') as f: f.write('call\\n')\n")
        result = self.run_cli(['--response-cache', str(cache)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(counter.read_text().splitlines()), 4)
        summary = json.loads(new_out.with_suffix('.summary.json').read_text())
        self.assertEqual(summary['imported_surrogate_responses'], 4)
        self.assertEqual(summary['fresh_surrogate_responses'], 0)
        self.assertEqual(summary['target_calls'], 0)
        result = self.run_cli(['--response-cache', str(cache), '--resume'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(counter.read_text().splitlines()), 4)

    def test_response_cache_rejects_changed_surrogate_temperature(self):
        self.assertEqual(self.run_cli().returncode, 0)
        cache = self.out.with_suffix('.trials.jsonl')
        self.command[self.command.index('--out') + 1] = str(self.out.with_name('changed.jsonl'))
        result = self.run_cli(['--response-cache', str(cache), '--surrogate-temperature', '0.5'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('surrogate_temperature mismatch', result.stderr)
        self.assertEqual(len(self.requests), 4)


if __name__ == "__main__":
    unittest.main()
