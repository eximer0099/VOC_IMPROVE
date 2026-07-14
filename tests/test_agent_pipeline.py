import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.critic import CriticAgent
from agents.evaluator import EvaluatorAgent
from agents.improver import PolicyImproverAgent
from agents.interpreter import NLInterpreterAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from llm_wrappers.anthropic_chat import AnthropicChat


TEST_CASES_PATH = (
    Path(__file__).resolve().parents[1] / "quality_diagnosis" / "test_cases.json"
)


def load_test_cases() -> list[dict]:
    cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise AssertionError(f"No test cases found in {TEST_CASES_PATH}")
    for index, case in enumerate(cases, start=1):
        if not str(case.get("question", "")).strip():
            raise AssertionError(f"Test case #{index} has no question")
    return cases


TEST_CASES = load_test_cases()


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    async def __call__(self, _prompt: str) -> str:
        return self.response


class FakePolicyLLM:
    def __init__(self, response: str = "", error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0
        self.prompts = []

    async def __call__(self, _prompt: str, **_kwargs) -> str:
        self.calls += 1
        self.prompts.append(_prompt)
        if self.error:
            raise self.error
        return self.response


class FakeInterpreterClient:
    def __init__(self, question: str, csv_path: str):
        self.responses = self
        self.question = question
        self.csv_path = csv_path

    async def create(self, **_kwargs):
        payload = json.dumps(
            {
                "task": "both",
                "filters": [self.question],
                "max_items": 30,
                "csv_path": self.csv_path,
            },
            ensure_ascii=False,
        )
        content = type("Content", (), {"text": payload})()
        output = type("Output", (), {"content": [content]})()
        return type("Response", (), {"output": [output]})()


class FakeCriticClient:
    def __init__(self):
        self.chat = self
        self.completions = self

    async def create(self, **_kwargs):
        message = type(
            "Message",
            (),
            {"content": '{"need_refine": false, "edits": [], "ask_more_samples": false}'},
        )()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class SixAgentPipelineTests(unittest.TestCase):
    EXPECTED_ORDER = [
        "Interpreter",
        "Retriever",
        "Summarizer",
        "Evaluator",
        "Critic",
        "Improver",
    ]

    def test_all_test_case_questions_run_through_six_agents_in_order(self):
        async def run_case(case: dict, csv_path: str):
            question = case["question"]
            order = []

            interpreter = NLInterpreterAgent.__new__(NLInterpreterAgent)
            interpreter.client = FakeInterpreterClient(question, csv_path)
            interpreter.retriever_endpoint = "unused-in-direct-agent-test"
            intent = await interpreter.parse(question, csv_path)
            order.append("Interpreter")

            retriever = RetrieverAgent()
            texts = await retriever.run(
                intent.csv_path, intent.filters, intent.max_items
            )
            order.append("Retriever")

            summarizer = SummarizerAgent.__new__(SummarizerAgent)
            summarizer.llm = FakeLLM(
                f"S0: {question} 관련 VOC 요약\n"
                f"S1: 고객 질문 분석: {question}\n"
                f"S2: 개선이 필요한 VOC: {question}"
            )
            candidates = await summarizer.make_candidates(
                texts, intent.max_items, n=3
            )
            order.append("Summarizer")

            evaluator = EvaluatorAgent.__new__(EvaluatorAgent)
            evaluator.llm = FakeLLM(
                '{"winner":"S1","scores":{"S0":8,"S1":9,"S2":7}}'
            )
            evaluation = await evaluator.evaluate(intent.task, candidates)
            summary = candidates[evaluation["winner"]]
            order.append("Evaluator")

            with patch("agents.critic.openai_client", FakeCriticClient()):
                criticism = await CriticAgent().review(summary, "summary")
            order.append("Critic")

            improver = PolicyImproverAgent.__new__(PolicyImproverAgent)
            improver.llm = FakePolicyLLM(
                f"정책 개선안: {question} 원인을 확인하고 담당 조직과 처리 기한을 지정합니다."
            )
            improver.fallback_llm = FakePolicyLLM("")
            policy = (await improver.improve(summary)).policy
            order.append("Improver")

            return {
                "order": order,
                "intent": intent,
                "texts": texts,
                "candidates": candidates,
                "evaluation": evaluation,
                "criticism": criticism,
                "summary": summary,
                "policy": policy,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = str(Path(temp_dir) / "voc.csv")
            for case in TEST_CASES:
                with self.subTest(case_id=case.get("case_id")):
                    with open(csv_path, "w", encoding="utf-8", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["고객ID", "불만내용"])
                        writer.writerow(["CUST001", case["question"]])

                    result = asyncio.run(run_case(case, csv_path))

                    self.assertEqual(result["order"], self.EXPECTED_ORDER)
                    self.assertEqual(result["intent"].task, "both")
                    self.assertTrue(result["texts"])
                    self.assertEqual(result["evaluation"]["winner"], "S1")
                    self.assertFalse(result["criticism"].need_refine)
                    self.assertIn(case["question"], result["summary"])
                    self.assertIn(case["question"], result["policy"])


class EvaluatorTests(unittest.TestCase):
    def test_numeric_winner_is_mapped_to_candidate_key(self):
        agent = EvaluatorAgent.__new__(EvaluatorAgent)
        agent.llm = FakeLLM(
            '{"winner":"3","scores":{"1":7.5,"2":8.0,"3":9.0}}'
        )
        candidates = {"S0": "first", "S1": "second", "S2": "third"}

        result = asyncio.run(agent.evaluate("both", candidates))

        self.assertEqual(result["winner"], "S2")
        self.assertEqual(result["scores"], {"S0": 7.5, "S1": 8.0, "S2": 9.0})
        self.assertEqual(candidates[result["winner"]], "third")

    def test_unknown_winner_falls_back_to_first_candidate(self):
        candidates = {"S0": "first", "S1": "second"}
        self.assertEqual(
            EvaluatorAgent._normalize_candidate_key("unknown", candidates), "S0"
        )


class SummarizerTests(unittest.TestCase):
    def test_multiline_candidates_are_preserved(self):
        agent = SummarizerAgent.__new__(SummarizerAgent)

        candidates = agent._parse_candidates(
            "S0: first line\ncontinued detail\nS1: another summary"
        )

        self.assertEqual(candidates["S0"], "first line\ncontinued detail")
        self.assertEqual(candidates["S1"], "another summary")


class AnthropicWrapperTests(unittest.TestCase):
    def test_all_text_blocks_are_joined(self):
        class Block:
            def __init__(self, text):
                self.text = text

        self.assertEqual(
            AnthropicChat._extract_text([Block("first"), Block("second")]),
            "first\nsecond",
        )


class ImproverTests(unittest.TestCase):
    def test_openai_fallback_uses_questions_from_test_cases(self):
        for case in TEST_CASES:
            with self.subTest(case_id=case.get("case_id")):
                question = case["question"]
                agent = PolicyImproverAgent.__new__(PolicyImproverAgent)
                agent.llm = FakePolicyLLM("")
                agent.fallback_llm = FakePolicyLLM(
                    "P0 정책 개선 담당 조직과 처리 기한을 명시하고 자동 복구 기능을 도입합니다."
                )

                result = asyncio.run(agent.improve(question))

                if len(question.strip()) < 10:
                    self.assertIn("요약 내용이", result.policy)
                    self.assertEqual(agent.llm.calls, 0)
                    self.assertEqual(agent.fallback_llm.calls, 0)
                else:
                    self.assertIn("자동 복구", result.policy)
                    self.assertEqual(agent.llm.calls, 1)
                    self.assertEqual(agent.fallback_llm.calls, 1)
                    self.assertIn(question, agent.llm.prompts[0])
                    self.assertIn(question, agent.fallback_llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
