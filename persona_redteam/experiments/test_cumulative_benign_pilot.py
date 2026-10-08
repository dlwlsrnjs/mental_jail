"""Offline checks for the fixed cumulative benign pilot."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.pcsa_judge import FIELDS
from experiments.cumulative_benign_pilot import ValidationError, run, validate_sequence
from perturbation.safe_surrogate_loop import METRICS, MODEL, SURROGATE_MODEL


def fixture():
    return json.loads((ROOT / "perturbation/examples/cumulative_medical_register_benign.json").read_text())


def completion(value, role):
    return {"task_role": role, "model": MODEL, "usage": {},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}


class CumulativePilotTest(unittest.TestCase):
    def test_sequence_is_one_edit_per_round_and_goal_is_unchanged(self):
        rounds = validate_sequence(fixture())
        self.assertEqual([r["round"] for r in rounds], [0, 1, 2])
        self.assertTrue(all(r["prompt"].endswith(fixture()["goal"]) for r in rounds))
        self.assertEqual(rounds[-1]["cumulative_medical_concepts"],
                         ["sleep-onset difficulty", "repetitive anticipatory worry"])
        changed = fixture(); changed["rounds"][2]["persona"]["condition"] = "New fact"
        with self.assertRaises(ValidationError):
            validate_sequence(changed)

    def test_mocked_pilot_records_real_call_counts_and_pcsa_gate(self):
        response = "I can help you make a safe and manageable plan."
        calls = []
        def fake_response(messages, model, **kwargs):
            calls.append(("surrogate", messages[1]["content"]))
            return {"text": response, "model": SURROGATE_MODEL, "finish_reason": "stop", "usage": {}}
        def fake_eval(body, role):
            calls.append((role, None))
            if role == "safe_quality_evaluator":
                metrics = {name: (0 if "penalty" in name else 4) for name in METRICS}
                return completion({"metrics": metrics, "recommendations": [], "reasoning": "Safe synthetic fixture."}, role)
            scores = {field: 0 for field in FIELDS}
            return completion({**scores, "evidence": [response], "reasoning": "Exact safe response evidence.",
                               "final_verdict": "SAFE"}, role)
        with tempfile.TemporaryDirectory() as temp:
            result = run(fixture(), "http://127.0.0.1:8014", Path(temp) / "result.json",
                         fake_response, fake_eval)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["summary"]["surrogate_calls"], 3)
        self.assertEqual(result["summary"]["pcsa_violation_counts"], [0, 0, 0])
        self.assertEqual(len(calls), 9)

    def test_invalid_pcsa_evidence_retries_without_repeating_surrogate(self):
        response = "I can help you make a safe and manageable plan."
        counts = {"surrogate": 0, "pcsa": 0}
        def fake_response(*args, **kwargs):
            counts["surrogate"] += 1
            return {"text": response, "model": SURROGATE_MODEL, "finish_reason": "stop", "usage": {}}
        def fake_eval(body, role):
            if role == "safe_quality_evaluator":
                metrics = {name: (0 if "penalty" in name else 4) for name in METRICS}
                return completion({"metrics": metrics, "recommendations": [], "reasoning": "Safe."}, role)
            counts["pcsa"] += 1
            scores = {field: 0 for field in FIELDS}
            evidence = "invented evidence" if counts["pcsa"] % 2 else response
            return completion({**scores, "evidence": [evidence], "reasoning": "Evidence check.",
                               "final_verdict": "SAFE"}, role)
        with tempfile.TemporaryDirectory() as temp:
            result = run(fixture(), "http://127.0.0.1:8014/v1", Path(temp) / "result.json",
                         fake_response, fake_eval)
        self.assertEqual(counts["surrogate"], 3)
        self.assertEqual(counts["pcsa"], 6)
        self.assertTrue(all(r["pcsa_validation_attempts"] == 2 for r in result["rounds"]))

    def test_resume_reuses_saved_surrogate_and_metrics(self):
        response = "I can help you make a safe and manageable plan."
        counts = {"surrogate": 0, "metric": 0, "pcsa": 0}
        def fake_response(*args, **kwargs):
            counts["surrogate"] += 1
            return {"text": response, "model": SURROGATE_MODEL, "finish_reason": "stop", "usage": {}}
        fail_once = {"value": True}
        def fake_eval(body, role):
            if role == "safe_quality_evaluator":
                counts["metric"] += 1
                metrics = {name: (0 if "penalty" in name else 4) for name in METRICS}
                return completion({"metrics": metrics, "recommendations": [], "reasoning": "Safe."}, role)
            counts["pcsa"] += 1
            if fail_once["value"]:
                raise ValidationError("temporary_test_failure", "stop after saved metrics")
            scores = {field: 0 for field in FIELDS}
            return completion({**scores, "evidence": [response], "reasoning": "Exact evidence.",
                               "final_verdict": "SAFE"}, role)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            with self.assertRaises(ValidationError):
                run(fixture(), "http://127.0.0.1:8014/v1", path, fake_response, fake_eval)
            self.assertEqual(json.loads(path.read_text())["rounds"][0]["status"], "metrics_saved")
            fail_once["value"] = False
            result = run(fixture(), "http://127.0.0.1:8014/v1", path, fake_response, fake_eval)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(counts["surrogate"], 3)
        self.assertEqual(counts["metric"], 3)


if __name__ == "__main__":
    unittest.main()
