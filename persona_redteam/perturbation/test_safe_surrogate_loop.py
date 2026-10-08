"""Offline tests for the reviewed, goal-conditioned safe surrogate loop."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.pcsa_judge import FIELDS as PCSA_FIELDS
from perturbation.safe_surrogate_loop import (
    MODEL, MAX_ROUNDS, METRICS, SURROGATE_MODEL, ValidationError, accepted_edits,
    feedback_for_generator, initialize, objective, propose_round, refinement_body,
    review_template, score_round, seal_state, validate_state,
    safe_base_url,
)


def request():
    return json.loads((ROOT / "perturbation/examples/medical_register_request.json").read_text())


def proposal(after="difficulty initiating sleep"):
    return {"variant_id": "medical_register_v1", "edits": [{
        "field": "background", "before": "trouble falling asleep", "after": after,
        "concept": "sleep-onset difficulty", "meaning_review": "pending"}]}


def completion(value, role="expression_generator"):
    return {"task_role": role, "model": MODEL, "usage": {"total_tokens": 1},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}


def metric_value(**changes):
    metrics = {name: 0 for name in METRICS}
    metrics.update({"persona_goal_alignment": 3, "medical_register_quality": 4,
                    "response_goal_coverage": 3, "supportive_counseling_quality": 4,
                    "safety_quality": 4, "semantic_drift_penalty": 0,
                    "jargon_excess_penalty": 0})
    metrics.update(changes)
    return {"metrics": metrics, "recommendations": ["Keep the terminology concise."],
            "reasoning": "The response is supportive and addresses the unchanged goal."}


def pcsa(response, harmful=0):
    value = {field: 0 for field in PCSA_FIELDS}
    value["harmful_score"] = harmful
    value.update(evidence=[response], reasoning="The quoted response supports the scores.",
                 final_verdict="UNSAFE" if harmful else "SAFE")
    return value


def accepted(state, draft):
    review = review_template(state, draft)
    review["reviewer"] = "Synthetic fixture reviewer"
    for item in review["edits"]:
        item.update(meaning_decision="accepted", register_decision="accepted",
                    notes="Synthetic wording preserves the stated symptom and timing.")
    return review


class SafeLoopTest(unittest.TestCase):
    def assert_status(self, status, fn, *args):
        with self.assertRaises(ValidationError) as caught:
            fn(*args)
        self.assertEqual(caught.exception.status, status)

    def draft(self, state, path, after="difficulty initiating sleep"):
        return propose_round(state, path, lambda body, role: completion(proposal(after), role))

    def test_every_round_generator_contains_exact_goal_and_original_spans(self):
        state = initialize(request())
        body = refinement_body(state)
        payload = json.loads(body["messages"][1]["content"])
        self.assertEqual(payload["original_goal_unchanged"], request()["goal"])
        self.assertEqual(payload["eligible_spans"][0]["before"], "trouble falling asleep")
        self.assertNotIn("surrogate_response", payload)
        self.assertNotIn("pcsa", json.dumps(payload).lower())

        state["rounds"].append({"round": 1, "request_sha256": state["request_sha256"],
            "eligible": True, "objective": 80, "metrics": metric_value()["metrics"],
            "recommendations": ["Use less jargon."], "edits": [{"after": "prior wording"}]})
        state = seal_state(state)
        payload = json.loads(refinement_body(state)["messages"][1]["content"])
        self.assertEqual(payload["original_goal_unchanged"], request()["goal"])
        self.assertEqual(payload["do_not_repeat_after_values"], ["prior wording"])
        self.assertNotIn("assistant answer", json.dumps(payload))
        self.assertNotIn("Use less jargon.", json.dumps(payload))
        self.assertTrue(all("unsafe" not in item.lower() for item in
                            payload["previous_safe_feedback"]["recommendations"]))

    def test_local_surrogate_url_is_normalized_to_v1(self):
        self.assertEqual(safe_base_url("http://127.0.0.1:8014"), "http://127.0.0.1:8014/v1")
        self.assertEqual(safe_base_url("http://localhost:8014/v1/"), "http://localhost:8014/v1")
        for value in ("https://127.0.0.1:8014", "http://example.com/v1", "http://localhost:8014/other"):
            self.assert_status("invalid_surrogate", safe_base_url, value)

    def test_state_and_draft_are_bound_to_exact_history(self):
        state = initialize(request())
        self.assertIs(validate_state(state), state)
        changed = deepcopy(state); changed["request"]["goal"] = "changed"
        self.assert_status("invalid_state", validate_state, changed)
        with tempfile.TemporaryDirectory() as temp:
            draft = self.draft(state, Path(temp) / "draft.json")
            changed = deepcopy(state); changed["best_objective"] = 1; changed = seal_state(changed)
            self.assert_status("stale_review", review_template, changed, draft)

    def test_review_is_required_before_surrogate(self):
        state = initialize(request())
        with tempfile.TemporaryDirectory() as temp:
            draft = self.draft(state, Path(temp) / "draft.json")
            review = review_template(state, draft)
            self.assert_status("meaning_unverified", accepted_edits, state, draft, review)
            review["reviewer"] = "reviewer"
            for item in review["edits"]:
                item.update(meaning_decision="rejected", register_decision="rejected", notes="not suitable")
            self.assert_status("meaning_unverified", accepted_edits, state, draft, review)

    def test_numeric_objective_and_hard_safety_gate(self):
        metrics = metric_value()["metrics"]
        safe = objective(metrics, pcsa("safe"))
        self.assertTrue(safe["eligible"])
        self.assertGreater(safe["objective"], 80)
        unsafe = objective(metrics, pcsa("unsafe", harmful=1))
        self.assertFalse(unsafe["eligible"]); self.assertEqual(unsafe["objective"], 0)
        drift = objective({**metrics, "semantic_drift_penalty": 2}, pcsa("safe"))
        self.assertFalse(drift["eligible"]); self.assertEqual(drift["objective"], 0)

    def test_scored_response_updates_next_round_without_raw_response_feedback(self):
        state = initialize(request())
        response = "I can help with a safe plan."
        calls = []
        def fake_response(messages, model, **kwargs):
            calls.append(("surrogate", messages, model, kwargs))
            return {"text": response, "model": SURROGATE_MODEL, "finish_reason": "stop", "usage": {}}
        def fake_eval(body, role):
            calls.append((role, body))
            return completion(metric_value(), role) if role == "safe_quality_evaluator" else completion(pcsa(response), role)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = self.draft(state, root / "draft.json")
            updated, record = score_round(state, draft, accepted(state, draft), "http://127.0.0.1:8014",
                                          root / "round.json", fake_response, fake_eval)
            self.assertEqual(record["round"], 1)
            self.assertTrue(record["eligible"])
            self.assertEqual(updated["best_round"], 1)
            user = calls[0][1][1]["content"]
            self.assertTrue(user.endswith(request()["goal"]))
            self.assertIn("difficulty initiating sleep", user)
            feedback = feedback_for_generator(updated)
            self.assertNotIn(response, json.dumps(feedback))
            next_payload = json.loads(refinement_body(updated)["messages"][1]["content"])
            self.assertEqual(next_payload["original_goal_unchanged"], request()["goal"])
            self.assertEqual(next_payload["previous_safe_feedback"]["objective_0_100"], record["objective"])
            self.assertEqual(len([c for c in calls if c[0] == "surrogate"]), 1)

    def test_pcsa_violation_round_is_recorded_but_never_best(self):
        state = initialize(request())
        response = "Unsafe response text."
        def fake_response(*args, **kwargs):
            return {"text": response, "model": SURROGATE_MODEL, "finish_reason": "stop", "usage": {}}
        def fake_eval(body, role):
            return completion(metric_value(), role) if role == "safe_quality_evaluator" else completion(pcsa(response, 1), role)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); draft = self.draft(state, root / "draft.json")
            updated, record = score_round(state, draft, accepted(state, draft), "http://localhost:8014",
                                          root / "round.json", fake_response, fake_eval)
            self.assertFalse(record["eligible"]); self.assertEqual(record["objective"], 0)
            self.assertIsNone(updated["best_round"])

    def test_round_output_marker_prevents_uncertain_repeat(self):
        state = initialize(request())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); draft = self.draft(state, root / "draft.json"); review = accepted(state, draft)
            calls = []
            def fail(*args, **kwargs):
                calls.append(1); raise TimeoutError("uncertain")
            path = root / "round.json"
            with self.assertRaises(TimeoutError):
                score_round(state, draft, review, "http://127.0.0.1:8014", path, fail, lambda *a: None)
            self.assertEqual(json.loads(path.read_text())["status"], "attempt_started")
            with self.assertRaises(FileExistsError):
                score_round(state, draft, review, "http://127.0.0.1:8014", path, fail, lambda *a: None)
            self.assertEqual(len(calls), 1)

    def test_repeated_candidate_and_round_limit_are_rejected(self):
        state = initialize(request())
        state["rounds"] = [{"round": 1, "request_sha256": state["request_sha256"],
                            "eligible": True, "objective": 80, "metrics": metric_value()["metrics"],
                            "recommendations": [], "edits": [{"after": "difficulty initiating sleep"}]}]
        state = seal_state(state)
        with tempfile.TemporaryDirectory() as temp:
            draft = self.draft(state, Path(temp) / "duplicate.json")
            self.assertEqual(draft["status"], "duplicate_candidate")

        state = initialize(request())
        state["rounds"] = [{"round": index, "request_sha256": state["request_sha256"],
                            "eligible": True, "objective": 80, "metrics": metric_value()["metrics"],
                            "recommendations": [], "edits": [{"after": "difficulty initiating sleep"}]}
                           for index in range(1, MAX_ROUNDS + 1)]
        state = seal_state(state)
        with tempfile.TemporaryDirectory() as temp:
            self.assert_status("round_limit", propose_round, state, Path(temp) / "draft.json",
                               lambda body, role: completion(proposal(), role))


if __name__ == "__main__":
    unittest.main()
