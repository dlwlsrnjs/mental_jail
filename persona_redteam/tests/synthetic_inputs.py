"""Build benign temporary inputs for routing and checkpoint tests.

Labels and risk fields exercise schema branches, not clinical annotation.
These examples must never enter research evaluation counts.
"""
import csv
import json
from pathlib import Path

LABELS = ("suicidal_ideation", "anxiety_crisis", "self-harm",
          "substance_abuse_or_withdrawal", "violent_thoughts", "risk_taking_behaviours")


def write_inputs(root):
    root = Path(root)
    for name in ("goals", "personas", "outputs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    vera = [{"Name": name, "Age": "30", "Gender": "unspecified",
             "Background": f"Fictional profile {name}: a person who enjoys reading.",
             "Short Current Suicide Risk Level": risk}
            for name, risk in (("fixture-v1", "High"), ("fixture-v2", "Immediate"),
                               ("fixture-v3", "High"), ("fixture-low", "Low"))]
    with (root / "personas/veramh_si_n100.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(vera[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(vera)
    cactus = [{"persona_id": f"fixture-c{i}", "descriptor": "A fictional student.",
               "background": "Interested in reading.", "condition": "Exam worry.",
               "thought": "I feel anxious about an exam.",
               "style_reference": "I speak in short sentences."} for i in range(1, 4)]
    goals, routed = [], []
    for index, label in enumerate(LABELS):
        pool = "VERA-SI" if label in {"suicidal_ideation", "self-harm"} else "Cactus"
        for is_request in (True, False):
            goal = {"goal_id": f"fixture-{index}-{int(is_request)}", "crisis_label": label,
                    "goal": f"Fictional case {index}, variant {int(is_request)}: I feel worried and want support.",
                    "is_client_utterance": True, "is_request": is_request}
            candidates = [{"source": pool, "id": f"fixture-{'v' if pool == 'VERA-SI' else 'c'}{i}",
                           "cos": 0.9 - 0.1 * i} for i in range(1, 4)]
            goals.append(goal)
            routed.append({**goal, "pathology": {"compliance_lever": "test-only failure marker"},
                           "routed_pool": pool, "persona_candidates": candidates})
    for path, rows in (("personas/cactus_distress_n2000.jsonl", cactus),
                       ("goals/crisis_goals_jmir_client.jsonl", goals),
                       ("outputs/goal_pathology_persona_routed_n813.jsonl", routed)):
        (root / path).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return root
