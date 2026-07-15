from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quality_diagnosis.qa_test_utils import (
    build_judge_cases_from_agent_log,
    cases_to_markdown,
)


class JudgeCaseLogMappingTests(unittest.TestCase):
    def test_missing_candidates_and_policy_are_written_as_json_null(self) -> None:
        events = [
            {"agent": "Interpreter", "action": "input", "question": "질문 1"},
            {"agent": "Interpreter", "action": "input", "question": "질문 2"},
            {
                "agent": "Summarizer",
                "action": "output",
                "operation": "make_candidates",
                "candidates": {"S0": "요약 2"},
            },
            {"agent": "Interpreter", "action": "input", "question": "질문 3"},
            {
                "agent": "Improver",
                "action": "output",
                "operation": "improve",
                "policy": "정책 3",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "agent.log"
            output_path = Path(temp_dir) / "judge_cases.json"
            log_path.write_text(
                "\n".join(json.dumps(event, ensure_ascii=False) for event in events),
                encoding="utf-8",
            )

            cases = build_judge_cases_from_agent_log(
                log_path, output_path, case_count=3
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(cases, saved)
        self.assertEqual(
            saved,
            [
                {"question": "질문 1", "candidates": None, "policy": None},
                {
                    "question": "질문 2",
                    "candidates": {"S0": "요약 2"},
                    "policy": None,
                },
                {"question": "질문 3", "candidates": None, "policy": "정책 3"},
            ],
        )
        markdown = cases_to_markdown(saved)
        self.assertGreaterEqual(markdown.count("NULL"), 3)


if __name__ == "__main__":
    unittest.main()
