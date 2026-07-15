import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from quality_diagnosis import llm_judge


class RubricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rubric = llm_judge.load_rubric(llm_judge.DEFAULT_RUBRIC)

    def test_rubric_has_expected_items_and_total_score(self) -> None:
        expected = set(llm_judge.OPENAI_ITEMS) | {llm_judge.ANTHROPIC_ITEM}

        self.assertEqual(set(self.rubric), expected)
        self.assertEqual(
            sum(item["max_score"] for item in self.rubric.values()),
            100,
        )

    def test_parse_max_score(self) -> None:
        self.assertEqual(llm_judge.parse_max_score("25점"), 25)
        self.assertEqual(llm_judge.parse_max_score(20), 20)

        with self.assertRaises(ValueError):
            llm_judge.parse_max_score("배점 없음")

    def test_validate_scores_rejects_missing_and_out_of_range_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "점수가 없습니다"):
            llm_judge.validate_scores({}, ("정확성",), self.rubric)

        with self.assertRaisesRegex(ValueError, "범위를 벗어났습니다"):
            llm_judge.validate_scores(
                {"정확성": 26},
                ("정확성",),
                self.rubric,
            )

    def test_deployment_decision_boundaries(self) -> None:
        self.assertEqual(llm_judge.deployment_decision(90), "배포 가능")
        self.assertEqual(llm_judge.deployment_decision(89.99), "조건부 배포")
        self.assertEqual(llm_judge.deployment_decision(80), "조건부 배포")
        self.assertEqual(llm_judge.deployment_decision(79.99), "개선 후 재시험")
        self.assertEqual(llm_judge.deployment_decision(70), "개선 후 재시험")
        self.assertEqual(llm_judge.deployment_decision(69.99), "배포 보류")
        self.assertEqual(
            llm_judge.deployment_decision(100, has_critical_defect=True),
            "배포 보류",
        )

    def test_critical_defect_forces_deployment_hold_in_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "defect_report.md"
            llm_judge.write_defect_report(
                report_path,
                average_score=95.0,
                case_count=20,
                critical_defects=[
                    {
                        "question": "환불이 완료되었나요?",
                        "type": "결제·환불 관련 잘못된 안내를 확정적으로 제공함",
                    }
                ],
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("평균 점수: 95.00점", report)
        self.assertIn("배포 여부: **배포 보류**", report)
        self.assertIn("결제·환불 관련 잘못된 안내를 확정적으로 제공함", report)
        self.assertIn("환불이 완료되었나요?", report)


class ProviderScoringTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rubric = llm_judge.load_rubric(llm_judge.DEFAULT_RUBRIC)
        cls.case = {
            "question": "로그인 인증번호가 늦게 도착합니다.",
            "candidates": {
                "S0": "배송 관련 요약",
                "S1": "인증번호 도착이 늦다는 요약",
                "S2": "로그인 관련 요약",
            },
            "policy": "인증번호 발송 지연을 모니터링하고 담당 조직이 개선한다.",
        }

    async def test_openai_policy_scores_are_parsed(self) -> None:
        expected = {
            "정확성": 22,
            "정책 구체성": 17,
            "유용성": 18,
            "안전성": 14,
            "critical_defects": ["존재하지 않는 정책·사실을 만들어 냄"],
        }
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        output_text=json.dumps(expected, ensure_ascii=False)
                    )
                )
            )
        )

        result = await llm_judge.score_policy_with_openai(
            client, self.case, self.rubric, "test-openai-model"
        )

        scores, defects = result
        self.assertEqual(
            scores,
            {
                key: float(expected[key])
                for key in llm_judge.OPENAI_ITEMS
            },
        )
        self.assertEqual(defects, ["존재하지 않는 정책·사실을 만들어 냄"])
        kwargs = client.responses.create.await_args.kwargs
        self.assertEqual(kwargs["model"], "test-openai-model")
        self.assertEqual(
            kwargs["text"]["format"]["schema"]["required"],
            [*llm_judge.OPENAI_ITEMS, "critical_defects"],
        )
        prompt = json.loads(kwargs["input"])
        self.assertEqual(prompt["question"], self.case["question"])
        self.assertEqual(prompt["policy"], self.case["policy"])

    async def test_anthropic_candidate_score_is_parsed(self) -> None:
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "content": [
                    {"type": "thinking", "text": "ignored"},
                    {
                        "type": "text",
                        "text": json.dumps({"요약 충실성": 16}, ensure_ascii=False),
                    },
                ]
            },
        )
        client = SimpleNamespace(post=AsyncMock(return_value=response))

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result = await llm_judge.score_candidates_with_anthropic(
                client, self.case, self.rubric, "test-anthropic-model"
            )

        self.assertEqual(result, {"요약 충실성": 16.0})
        args, kwargs = client.post.await_args
        self.assertEqual(args[0], "https://api.anthropic.com/v1/messages")
        self.assertEqual(kwargs["headers"]["x-api-key"], "test-key")
        self.assertEqual(kwargs["json"]["model"], "test-anthropic-model")
        self.assertEqual(kwargs["json"]["max_tokens"], 512)
        self.assertEqual(kwargs["json"]["thinking"], {"type": "disabled"})
        score_definition = kwargs["json"]["output_config"]["format"]["schema"][
            "properties"
        ]["요약 충실성"]
        self.assertEqual(score_definition, {"type": "number"})
        prompt = json.loads(kwargs["json"]["messages"][0]["content"])
        self.assertEqual(prompt["question"], self.case["question"])
        self.assertEqual(prompt["candidates"], self.case["candidates"])


class RunJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_row_is_saved_before_a_later_case_fails(self) -> None:
        cases = [
            {"question": "완료 케이스", "candidates": {"S0": "요약"}, "policy": "정책"},
            {"question": "실패 케이스", "candidates": {"S0": "요약"}, "policy": "정책"},
        ]
        policy_score = {
            "정확성": 20.0,
            "정책 구체성": 15.0,
            "유용성": 15.0,
            "안전성": 10.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            cases_path = temp_dir / "cases.json"
            output_path = temp_dir / "result.csv"
            cases_path.write_text(
                json.dumps(cases, ensure_ascii=False), encoding="utf-8"
            )

            with (
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "test-openai-key",
                        "ANTHROPIC_API_KEY": "test-anthropic-key",
                    },
                    clear=True,
                ),
                patch.object(llm_judge, "AsyncOpenAI", return_value=object()),
                patch.object(
                    llm_judge,
                    "score_policy_with_openai",
                    new=AsyncMock(
                        side_effect=[(policy_score, []), RuntimeError("API 실패")]
                    ),
                ),
                patch.object(
                    llm_judge,
                    "score_candidates_with_anthropic",
                    new=AsyncMock(
                        side_effect=[{"요약 충실성": 20.0}, {"요약 충실성": 20.0}]
                    ),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "API 실패"):
                    await llm_judge.run_judge(
                        cases_path,
                        llm_judge.DEFAULT_RUBRIC,
                        output_path,
                        defect_report_path=temp_dir / "defect_report.md",
                    )

            with output_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            json_rows = json.loads(
                output_path.with_suffix(".json").read_text(encoding="utf-8")
            )
            self.assertFalse((temp_dir / "defect_report.md").exists())

        self.assertEqual(
            rows,
            [{"question": "완료 케이스", "총점": "80.0", "PASS/FAIL": "PASS"}],
        )
        self.assertEqual(len(json_rows), 1)
        self.assertEqual(json_rows[0]["question"], "완료 케이스")
        self.assertEqual(
            set(json_rows[0]),
            {
                "question",
                "정확성",
                "정책 구체성",
                "유용성",
                "안전성",
                "요약 충실성",
            },
        )

    async def test_run_judge_writes_three_column_csv_and_pass_fail_boundary(self) -> None:
        cases = [
            {"question": "80점 케이스", "candidates": {"S0": "요약"}, "policy": "정책"},
            {"question": "79점 케이스", "candidates": {"S0": "요약"}, "policy": "정책"},
        ]
        policy_results = [
            {"정확성": 20.0, "정책 구체성": 15.0, "유용성": 15.0, "안전성": 10.0},
            {"정확성": 20.0, "정책 구체성": 15.0, "유용성": 14.0, "안전성": 10.0},
        ]
        candidate_results = [{"요약 충실성": 20.0}, {"요약 충실성": 20.0}]

        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            cases_path = temp_dir / "cases.json"
            output_path = temp_dir / "reports" / "result.csv"
            report_path = temp_dir / "defect_report.md"
            cases_path.write_text(
                json.dumps(cases, ensure_ascii=False), encoding="utf-8"
            )

            with (
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "test-openai-key",
                        "ANTHROPIC_API_KEY": "test-anthropic-key",
                    },
                    clear=True,
                ),
                patch.object(llm_judge, "AsyncOpenAI", return_value=object()),
                patch.object(
                    llm_judge,
                    "score_policy_with_openai",
                    new=AsyncMock(
                        side_effect=[(scores, []) for scores in policy_results]
                    ),
                ),
                patch.object(
                    llm_judge,
                    "score_candidates_with_anthropic",
                    new=AsyncMock(side_effect=candidate_results),
                ),
            ):
                await llm_judge.run_judge(
                    cases_path,
                    llm_judge.DEFAULT_RUBRIC,
                    output_path,
                    defect_report_path=report_path,
                )

            with output_path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            json_rows = json.loads(
                output_path.with_suffix(".json").read_text(encoding="utf-8")
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(reader.fieldnames, ["question", "총점", "PASS/FAIL"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"question": "80점 케이스", "총점": "80.0", "PASS/FAIL": "PASS"})
        self.assertEqual(rows[1], {"question": "79점 케이스", "총점": "79.0", "PASS/FAIL": "FAIL"})
        self.assertEqual(len(json_rows), 2)
        self.assertEqual(json_rows[0]["정확성"], 20.0)
        self.assertEqual(json_rows[0]["요약 충실성"], 20.0)
        self.assertIn("평균 점수: 79.50점", report)
        self.assertIn("배포 여부: **개선 후 재시험**", report)

    async def test_run_judge_rejects_case_with_missing_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            cases_path = temp_dir / "cases.json"
            cases_path.write_text(
                json.dumps([{"question": "질문", "policy": "정책"}], ensure_ascii=False),
                encoding="utf-8",
            )

            with (
                patch.dict(
                    "os.environ",
                    {
                        "OPENAI_API_KEY": "test-openai-key",
                        "ANTHROPIC_API_KEY": "test-anthropic-key",
                    },
                    clear=True,
                ),
                patch.object(llm_judge, "AsyncOpenAI", return_value=object()),
            ):
                with self.assertRaisesRegex(ValueError, "candidates"):
                    await llm_judge.run_judge(
                        cases_path,
                        llm_judge.DEFAULT_RUBRIC,
                        temp_dir / "result.csv",
                        defect_report_path=temp_dir / "defect_report.md",
                    )


if __name__ == "__main__":
    unittest.main()
