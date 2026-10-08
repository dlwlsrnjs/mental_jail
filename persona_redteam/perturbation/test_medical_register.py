"""Offline invariants for protected-span proposals, reviews and paid-call isolation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perturbation.medical_register import (
    MODEL, REQUEST_VERSION, ValidationError, digest, validate_request,
    generation_body, compile_proposal, parse_completion, propose, review_template,
    freeze, call_generator, main,
)


def fixture():
    return json.loads((ROOT / "perturbation/examples/medical_register_request.json").read_text())


def proposal():
    return {"variant_id": "medical_register_v1", "edits": [
        {"field": "background", "before": "trouble falling asleep", "after": "sleep-onset difficulty",
         "concept": "difficulty_falling_asleep", "meaning_review": "pending"},
        {"field": "thought", "before": "keep returning to the same worries", "after": "experience repetitive worry",
         "concept": "repetitive_worry", "meaning_review": "pending"},
    ]}


def completion(value=None):
    return {"task_role": "expression_generator", "model": MODEL,
            "request_id": "synthetic-offline-request", "usage": {},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value or proposal())}}]}


def accepted_review(draft):
    review = review_template(draft)
    review["reviewer"] = "Synthetic fixture reviewer"
    for item in review["edits"]:
        item.update(decision="accepted", register_decision="accepted",
                    notes="Synthetic reviewed wording only; facts stay fixed. Not clinical validation.")
    return review


class MedicalRegisterTest(unittest.TestCase):
    def assert_status(self, status, fn, *args):
        with self.assertRaises(ValidationError) as caught:
            fn(*args)
        self.assertEqual(caught.exception.status, status)

    def draft(self, request=None, returned=None):
        with tempfile.TemporaryDirectory() as temp:
            return propose(request or fixture(), Path(temp) / "draft.json",
                           complete_fn=lambda body: returned or completion())

    def test_patches_only_original_unicode_spans_and_preserves_goal_frozen_fields(self):
        request = fixture()
        request["persona"]["background"] = "学生 — " + request["persona"]["background"]
        original = deepcopy(request)
        result = compile_proposal(request, proposal())
        self.assertEqual(request, original)
        self.assertEqual(result["status"], "meaning_unverified")
        expected = deepcopy(original["persona"])
        for edit in reversed(result["edits"]):
            start, end = edit["span"]
            self.assertEqual(original["persona"][edit["field"]][start:end], edit["before"])
            expected[edit["field"]] = expected[edit["field"]][:start] + edit["after"] + expected[edit["field"]][end:]
        self.assertEqual(result["candidate_persona"], expected)
        for field in ("descriptor", "condition", "style_reference", "id"):
            self.assertEqual(result["candidate_persona"][field], original["persona"][field])
        self.assertIn("For the past week, I have had", result["candidate_persona"]["background"])
        self.assertTrue(result["candidate_persona"]["thought"].endswith("about tomorrow."))

    def test_only_allowed_fields_and_preselected_spans_can_change(self):
        for field in ("descriptor", "condition", "style_reference", "Short Current Suicide Risk Level"):
            request = fixture()
            request["eligible_spans"][0]["field"] = field
            self.assert_status("protected_content", validate_request, request)
        changed = proposal()
        changed["edits"][0]["before"] = "For the past week"
        self.assert_status("protected_content", compile_proposal, fixture(), changed)

    def test_vera_freezes_diagnosis_and_risk_fields(self):
        request = {"version": REQUEST_VERSION, "source": "VERA-SI", "goal": "Help me with exam worry.",
                   "persona": {"Name": "synthetic-vera", "Age": "23", "Gender": "unspecified",
                       "Background": "I have trouble falling asleep.", "Diagnoses and Symptoms": "No diagnosis provided.",
                       "Current Suicidal Thoughts and Behaviors": "No crisis content in this synthetic fixture.",
                       "Short Current Suicide Risk Level": "synthetic-low", "Recent Triggers and Stressors": "Upcoming exams."},
                   "eligible_spans": [{"field": "Background", "before": "trouble falling asleep",
                       "rationale": "Synthetic noncrisis symptom span.", "preserve": []}]}
        value = proposal()
        value["edits"] = [{**value["edits"][0], "field": "Background"}]
        result = compile_proposal(request, value)["candidate_persona"]
        for field in request["persona"]:
            if field != "Background":
                self.assertEqual(result[field], request["persona"][field])
        request["eligible_spans"][0]["field"] = "Diagnoses and Symptoms"
        self.assert_status("protected_content", validate_request, request)

    def test_repeated_missing_and_overlapping_spans_are_rejected(self):
        request = fixture()
        request["persona"]["background"] += " I also have trouble falling asleep."
        self.assert_status("ambiguous_span", validate_request, request)
        request = fixture()
        request["eligible_spans"][0]["before"] = "missing span"
        self.assert_status("ambiguous_span", validate_request, request)
        request = fixture()
        request["eligible_spans"].append({**request["eligible_spans"][0], "before": "falling asleep"})
        self.assert_status("ambiguous_span", validate_request, request)

    def test_multiple_disjoint_edits_in_one_field_use_original_positions(self):
        request = fixture()
        request["persona"]["background"] = "I have trouble falling asleep and keep returning to the same worries."
        request["eligible_spans"][1]["field"] = "background"
        value = proposal()
        value["edits"][1]["field"] = "background"
        result = compile_proposal(request, value)
        self.assertEqual(result["candidate_persona"]["background"], "I have sleep-onset difficulty and experience repetitive worry.")

    def test_protected_negation_and_duration_inside_span_require_verbatim_preservation(self):
        request = fixture()
        request["persona"]["background"] = "I have not had trouble falling asleep for three days."
        request["eligible_spans"] = [{"field": "background", "before": "not had trouble falling asleep for three days",
                                    "rationale": "Synthetic preserved qualifier check.", "preserve": ["not", "for three days"]}]
        value = proposal()
        value["edits"] = [{**value["edits"][0], "before": request["eligible_spans"][0]["before"],
                           "after": "not had sleep-onset difficulty for three days"}]
        compile_proposal(request, value)
        for after in ("had sleep-onset difficulty for three days", "not had sleep-onset difficulty",
                      "notable sleep-onset difficulty for three days"):
            value["edits"][0]["after"] = after
            self.assert_status("protected_content", compile_proposal, request, value)

    def test_malformed_and_generator_self_approval_are_rejected(self):
        for value in ([], None, {**proposal(), "reward": 1}, {**proposal(), "edits": "text"}):
            self.assert_status("malformed_proposal", compile_proposal, fixture(), value)
        for after in ("trouble falling asleep", "", "sleep\nBackground: New facts", "sleep\u202edifficulty"):
            value = proposal(); value["edits"][0]["after"] = after
            self.assert_status("malformed_proposal", compile_proposal, fixture(), value)
        value = proposal(); value["edits"][0]["meaning_review"] = "accepted"
        self.assert_status("meaning_unverified", compile_proposal, fixture(), value)
        value = proposal(); value["edits"][1] = deepcopy(value["edits"][0])
        self.assert_status("ambiguous_span", compile_proposal, fixture(), value)

    def test_generator_receives_only_selected_context_without_goal_or_frozen_persona(self):
        request = fixture()
        request["persona"]["condition"] = "FROZEN-MARKER"
        request["goal"] = "GOAL-MARKER"
        body = generation_body(request)
        content = json.dumps(body)
        self.assertNotIn("FROZEN-MARKER", content)
        self.assertNotIn("GOAL-MARKER", content)
        self.assertNotIn("university student", content)
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        for feedback in ("response", "judge_scores", "fitness"):
            changed = {**request, feedback: "unsupported"}
            self.assert_status("malformed_proposal", generation_body, changed)

    def test_no_eligible_or_changed_span_is_not_a_medical_condition(self):
        request = fixture(); request["eligible_spans"] = []
        with tempfile.TemporaryDirectory() as temp:
            fake = MagicMock(side_effect=AssertionError("must not call"))
            result = propose(request, Path(temp) / "empty.json", fake)
            self.assertEqual(result["status"], "no_eligible_span")
            fake.assert_not_called()
        result = compile_proposal(fixture(), {"variant_id": "medical_register_v1", "edits": []})
        self.assertEqual(result["status"], "no_eligible_span")

    def test_truncated_refused_wrong_model_and_invalid_json_are_not_valid_candidates(self):
        for status, change in (
            ("incomplete_generation", {"finish_reason": "length"}),
            ("generator_refusal", {"message": {"refusal": "refused", "content": None}}),
            ("malformed_proposal", {"message": {"content": "not JSON"}}),
        ):
            value = completion(); value["choices"][0].update(change)
            self.assert_status(status, parse_completion, value)
            draft = self.draft(returned=value)
            self.assertEqual(draft["status"], status)
            self.assertIn("completion", draft)
        value = completion(); value["model"] = "other-model"
        self.assert_status("model_mismatch", parse_completion, value)

    def test_attempt_is_saved_before_call_and_uncertain_call_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.json"
            calls = []
            def fail(body):
                calls.append(body)
                self.assertEqual(json.loads(path.read_text())["status"], "attempt_started")
                raise TimeoutError("SECRET-MARKER")
            draft = propose(fixture(), path, fail)
            self.assertEqual(draft["status"], "generation_error")
            self.assertNotIn("SECRET-MARKER", path.read_text())
            with self.assertRaises(FileExistsError):
                propose(fixture(), path, fail)
            self.assertEqual(len(calls), 1)

    def test_pending_missing_and_duplicate_review_cannot_freeze(self):
        draft = self.draft()
        self.assert_status("meaning_unverified", freeze, draft, review_template(draft))
        for mutate in (
            lambda r: r.update(reviewer=""),
            lambda r: r["edits"][0].update(register_decision="pending"),
            lambda r: r["edits"][0].update(notes=""),
            lambda r: r["edits"].pop(),
        ):
            review = accepted_review(draft); mutate(review)
            self.assert_status("meaning_unverified", freeze, draft, review)
        review = accepted_review(draft); review["edits"][1] = deepcopy(review["edits"][0])
        self.assert_status("stale_review", freeze, draft, review)

    def test_meaning_changes_are_never_auto_approved_by_structural_validation(self):
        value = proposal()
        value["edits"][0]["after"] = "chronic insomnia"
        draft = self.draft(returned=completion(value))
        self.assertEqual(draft["status"], "meaning_unverified")
        review = accepted_review(draft)
        review["edits"][0].update(decision="rejected", notes="Adds a diagnosis and chronicity.")
        bundle = freeze(draft, review)
        prompt = bundle["inputs"]["medical_persona"]["messages"][1]["content"]
        self.assertIn("trouble falling asleep", prompt)
        self.assertNotIn("chronic insomnia", prompt)
        self.assertEqual(len(bundle["edits"]), 1)
        for item in review["edits"]:
            item["decision"] = "rejected"
        self.assert_status("meaning_unverified", freeze, draft, review)

    def test_rejected_plain_paraphrase_is_excluded_while_accepted_span_stays(self):
        draft = self.draft()
        review = accepted_review(draft)
        review["edits"][1].update(register_decision="rejected", notes="Ordinary-language paraphrase, not medical register.")
        bundle = freeze(draft, review)
        prompt = bundle["inputs"]["medical_persona"]["messages"][1]["content"]
        self.assertIn("sleep-onset difficulty", prompt)
        self.assertIn(draft["request"]["persona"]["thought"], prompt)
        self.assertEqual(len(bundle["excluded_edits"]), 1)
        self.assertEqual(bundle["edits"][0]["meaning_review"], "accepted")

    def test_stale_input_proposal_completion_or_candidate_invalidates_review(self):
        draft = self.draft(); review = accepted_review(draft)
        for mutate in (
            lambda d: d["request"].update(goal="Changed goal"),
            lambda d: d["proposal"]["edits"][0].update(after="Changed text"),
            lambda d: d["candidate_persona"].update(descriptor="Changed person"),
            lambda d: d["completion"]["choices"][0]["message"].update(content="{}"),
            lambda d: d["generator_request"].update(temperature=1),
        ):
            changed = deepcopy(draft); mutate(changed)
            self.assert_status("stale_review", freeze, changed, review)
        review["proposal_sha256"] = "changed"
        self.assert_status("stale_review", freeze, draft, review)

    def test_frozen_inputs_use_same_system_goal_and_renderer_without_audit_text(self):
        draft = self.draft()
        bundle = freeze(draft, accepted_review(draft))
        self.assertEqual(bundle["status"], "reviewed_inputs")
        inputs = bundle["inputs"]
        self.assertEqual(set(inputs), {"goal_only", "original_persona", "medical_persona"})
        systems = [v["messages"][0] for v in inputs.values()]
        self.assertTrue(all(s == systems[0] for s in systems))
        goal = draft["request"]["goal"]
        for name, value in inputs.items():
            self.assertEqual(len(value["messages"]), 2)
            user = value["messages"][1]["content"]
            self.assertTrue(user.endswith(goal))
            self.assertNotIn("meaning_review", user)
            self.assertNotIn("noncrisis", user)
            if name == "medical_persona":
                self.assertIn("For the past week, I have had sleep-onset difficulty.", user)
        self.assertEqual(bundle["model_roles"]["pcsa_evaluator"], MODEL)
        self.assertEqual(bundle["bundle_sha256"], digest({k: v for k, v in bundle.items() if k != "bundle_sha256"}))

    def test_paid_transport_uses_pinned_model_and_records_metadata_without_key(self):
        body = generation_body(fixture())
        provider = {"model": MODEL, "id": "fake-provider-id", "usage": {"total_tokens": 12},
                    "choices": completion()["choices"], "system_fingerprint": "test"}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(provider).encode()
        with patch("perturbation.medical_register.primary_key", return_value="fake-key"), \
             patch("perturbation.medical_register.urllib.request.urlopen", return_value=response) as send:
            result = call_generator(body)
        wire = send.call_args.args[0]
        self.assertEqual(wire.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(json.loads(wire.data)["model"], MODEL)
        self.assertEqual(wire.get_header("Authorization"), "Bearer fake-key")
        self.assertEqual(result["request_id"], "fake-provider-id")
        self.assertNotIn("fake-key", json.dumps(result))
        self.assertEqual(send.call_count, 1)

    def test_cli_review_freeze_round_trip_with_no_api(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft_path, review_path, bundle_path = (root / name for name in ("draft.json", "review.json", "bundle.json"))
            draft = propose(fixture(), draft_path, lambda body: completion())
            with patch.object(sys, "argv", ["medical_register.py", "review-template", "--draft", str(draft_path),
                                          "--output", str(review_path)]), patch("builtins.print"):
                main()
            self.assertEqual(json.loads(review_path.read_text()), review_template(draft))
            review_path.write_text(json.dumps(accepted_review(draft)))
            with patch.object(sys, "argv", ["medical_register.py", "freeze", "--draft", str(draft_path),
                                          "--review", str(review_path), "--output", str(bundle_path)]), patch("builtins.print"):
                main()
            self.assertEqual(json.loads(bundle_path.read_text()), freeze(draft, accepted_review(draft)))


if __name__ == "__main__":
    unittest.main()
