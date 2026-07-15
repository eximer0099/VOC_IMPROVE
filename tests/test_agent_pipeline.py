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
from grpc_server import VOCGRPCRuntime
from llm_wrappers.anthropic_chat import AnthropicChat
from utils.text_normalization import build_search_terms, normalize_korean_text


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
    def __init__(self, response=None):
        self.chat = self
        self.completions = self
        self.response = response or {
            "need_refine": False,
            "edits": [],
            "ask_more_samples": False,
        }
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type(
            "Message",
            (),
            {"content": json.dumps(self.response, ensure_ascii=False)},
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


class InterpreterClarificationTests(unittest.TestCase):
    def test_short_generic_question_requires_clarification_without_history(self):
        self.assertTrue(NLInterpreterAgent.is_ambiguous_question("왜 안돼요?"))

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = str(Path(temp_dir) / "voc.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows(
                    [["고객ID", "불만내용"], ["CUST001", "배송이 지연됩니다."]]
                )

            history = NLInterpreterAgent.find_similar_voc_history(
                "왜 안돼요?", csv_path
            )

        self.assertEqual(history, [])

    def test_short_question_is_enriched_from_similar_voc_history(self):
        self.assertTrue(NLInterpreterAgent.is_ambiguous_question("환불 안돼요"))

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = str(Path(temp_dir) / "voc.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows(
                    [
                        ["고객ID", "불만내용"],
                        ["CUST001", "환불 신청 후 처리 상태를 알 수 없습니다."],
                        ["CUST002", "배송 주소를 변경할 수 없습니다."],
                    ]
                )

            history = NLInterpreterAgent.find_similar_voc_history(
                "환불 안돼요", csv_path
            )

        self.assertEqual(len(history), 1)
        self.assertIn("환불", history[0])


class SearchNormalizationTests(unittest.TestCase):
    def test_typo_and_colloquial_expression_are_normalized(self):
        normalized = normalize_korean_text("결제됫는대 주문안보여요 ㅠㅠ")
        terms = build_search_terms(["결제됫는대 주문안보여요 ㅠㅠ"])

        self.assertEqual(normalized, "결제됐는데 주문 보이지 않아요")
        self.assertIn("주문", terms)
        self.assertIn("보이지", terms)
        self.assertNotIn("않아요", terms)

    def test_retriever_matches_standard_voc_from_typo_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = str(Path(temp_dir) / "voc.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows(
                    [
                        ["고객ID", "불만내용"],
                        ["CUST001", "결제 후 주문 내역이 보이지 않습니다."],
                        ["CUST002", "배송 주소를 변경할 수 없습니다."],
                    ]
                )

            results = asyncio.run(
                RetrieverAgent().run(
                    csv_path, ["결제됫는대 주문안보여요"], max_items=10
                )
            )

        self.assertEqual(len(results), 1)
        self.assertIn("CUST001", results[0])

    def test_colloquial_failure_term_matches_formal_voc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = str(Path(temp_dir) / "voc.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows(
                    [["고객ID", "불만내용"], ["CUST001", "앱이 작동하지 않습니다."]]
                )

            results = asyncio.run(
                RetrieverAgent().run(csv_path, ["앱 먹통"], max_items=10)
            )

        self.assertEqual(len(results), 1)
        self.assertIn("작동하지", results[0])


class SummarizerTests(unittest.TestCase):
    def test_multiline_candidates_are_preserved(self):
        agent = SummarizerAgent.__new__(SummarizerAgent)

        candidates = agent._parse_candidates(
            "S0: first line\ncontinued detail\nS1: another summary"
        )

        self.assertEqual(candidates["S0"], "first line\ncontinued detail")
        self.assertEqual(candidates["S1"], "another summary")

    def test_grounding_check_keeps_only_exact_source_citations(self):
        texts = [
            "CUST001 배송이 지연되고 안내를 받지 못했습니다.",
            "CUST002 결제 후 주문 내역이 보이지 않습니다.",
        ]
        candidates = {
            "S0": (
                "배송이 지연되고 안내를 받지 못했습니다. | 근거: [VOC1] "
                "CUST001 배송이 지연되고 안내를 받지 못했습니다."
            ),
            "S1": (
                "서버 장애로 3일 지연되었습니다. | 근거: [VOC2] "
                "CUST002 결제 후 주문 내역이 보이지 않습니다."
            ),
        }

        checked = SummarizerAgent.grounding_check(candidates, texts, 2)

        self.assertEqual(checked["S0"], candidates["S0"])
        self.assertNotEqual(checked["S1"], candidates["S1"])
        self.assertIn(texts[1], checked["S1"])
        self.assertNotIn("서버 장애", checked["S1"])
        self.assertTrue(
            all(
                SummarizerAgent._is_grounded_candidate(candidate, texts)
                for candidate in checked.values()
            )
        )

    def test_make_candidates_prompt_requires_evidence_and_replaces_hallucination(self):
        class CapturingLLM:
            def __init__(self):
                self.prompt = ""

            async def __call__(self, prompt):
                self.prompt = prompt
                return "S0: 시스템 장애가 원인입니다. | 근거: [VOC1] 조작된 원문"

        texts = ["CUST001 로그인 인증번호가 늦게 도착합니다."]
        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent.llm = CapturingLLM()

        candidates = asyncio.run(agent.make_candidates(texts, max_items=1, n=1))

        self.assertIn("원문에 없는 사실", agent.llm.prompt)
        self.assertIn("[VOC1]", agent.llm.prompt)
        self.assertTrue(
            SummarizerAgent._is_grounded_candidate(candidates["S0"], texts)
        )
        self.assertNotIn("시스템 장애", candidates["S0"])

    def test_refine_rejects_result_not_grounded_in_source(self):
        source = "CUST001 환불 처리 상태를 확인할 수 없습니다."
        draft = f"환불 처리 상태 | 근거: [VOC1] {source}"
        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent.llm = FakeLLM(
            "환불이 완료되었습니다. | 근거: [VOC1] " + source
        )

        refined = asyncio.run(
            agent.refine(draft, '{"edits": ["구체화"]}', source_texts=[source])
        )

        self.assertEqual(refined, draft)


class CriticTests(unittest.TestCase):
    def test_summary_review_does_not_request_facts_absent_from_voc(self):
        client = FakeCriticClient()

        with patch("agents.critic.openai_client", client):
            asyncio.run(
                CriticAgent().review(
                    "결제 후 주문 내역 미노출 | 근거: [VOC1] 결제 후 주문 내역 미노출",
                    "summary",
                )
            )

        prompt = client.calls[0]["messages"][1]["content"]
        self.assertIn("근거에 없는 시점", prompt)
        self.assertIn("원인 가설", prompt)
        self.assertIn("ask_more_samples=true", prompt)

    def test_revalidation_checks_previous_edits_against_revised_summary(self):
        client = FakeCriticClient(
            {
                "need_refine": True,
                "edits": ["담당 조직 명시는 아직 반영되지 않음"],
                "ask_more_samples": False,
            }
        )
        document = json.dumps(
            {
                "previous_edits": ["담당 조직을 명시해라"],
                "revised_summary": "배송 지연 문의가 반복됨",
            },
            ensure_ascii=False,
        )

        with patch("agents.critic.openai_client", client):
            result = asyncio.run(
                CriticAgent().review(document, "summary_revalidation")
            )

        prompt = client.calls[0]["messages"][1]["content"]
        self.assertIn("previous_edits", prompt)
        self.assertIn("revised_summary", prompt)
        self.assertIn("실제로 반영", prompt)
        self.assertTrue(result.need_refine)
        self.assertEqual(
            result.edits, ["담당 조직 명시는 아직 반영되지 않음"]
        )


class IntentTopicGuardrailTests(unittest.TestCase):
    def test_matching_intent_topic_passes(self):
        intent = {
            "task": "both",
            "filters": ["배송", "지연"],
            "max_items": 20,
            "csv_path": "voc.csv",
        }

        checked = VOCGRPCRuntime._apply_intent_topic_guardrail(
            intent,
            {
                "ok": True,
                "summary": "배송 지연 문의가 반복됩니다.",
                "policy": "예상 도착 알림을 제공합니다.",
                "trace": "pipeline_completed",
            },
        )

        self.assertTrue(checked["ok"])
        self.assertTrue(json.loads(checked["intent_guardrail_json"])["passed"])

    def test_off_topic_response_is_blocked_and_cleared(self):
        intent = {
            "task": "both",
            "filters": ["로그인 인증번호"],
            "max_items": 20,
            "csv_path": "voc.csv",
        }

        checked = VOCGRPCRuntime._apply_intent_topic_guardrail(
            intent,
            {
                "ok": True,
                "summary": "배송 주소 변경 문의가 반복됩니다.",
                "policy": "택배 알림을 강화합니다.",
                "trace": "pipeline_completed",
            },
        )

        guardrail = json.loads(checked["intent_guardrail_json"])
        self.assertFalse(checked["ok"])
        self.assertFalse(guardrail["passed"])
        self.assertEqual(checked["summary"], "")
        self.assertEqual(checked["policy"], "")
        self.assertIn("intent_topic_guardrail_failed", checked["trace"])

    def test_requested_output_type_must_exist(self):
        guardrail = VOCGRPCRuntime.validate_intent_topic(
            {"task": "both", "filters": ["환불"]},
            "환불 문의가 있습니다.",
            "",
        )

        self.assertFalse(guardrail["passed"])
        self.assertEqual(guardrail["missing_outputs"], ["policy"])


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
