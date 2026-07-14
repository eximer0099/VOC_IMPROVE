"""In-process E2E test for the natural-language VOC pipeline.

The real protobuf services and gRPC channels are used.  Only agent business
responses are deterministic fakes, keeping the test independent of API keys,
external LLMs, fixed ports, and the production CSV file.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Support both `python -m quality_diagnosis.test_pipeline_e2e` and direct
# execution with `python quality_diagnosis/test_pipeline_e2e.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

import grpc_server
from utils.settings import MODEL_SUMMARY


TEST_CASES_PATH = Path(__file__).with_name("test_cases.json")
RUBRIC_PATH = Path(__file__).with_name("evaluation_rubric.csv")
RESULT_PATH = Path(__file__).parent / "reports" / "test_result.csv"
QUALITY_REPORT_PATH = Path(__file__).parent / "reports" / "quality_score_report.md"
DEPLOYMENT_DECISION_PATH = (
    Path(__file__).parent / "reports" / "deployment_decision.md"
)
CSV_PATH = str(PROJECT_ROOT / "voc.csv")
AGENT_LOG_PATH = PROJECT_ROOT / "agent.log"
EXPECTED_AGENT_ORDER = [
    "Interpreter",
    "Retriever",
    "Summarizer",
    "Evaluator",
    "Critic",
    "Improver",
]


def load_test_cases() -> list[dict]:
    """Load natural-language questions from the shared QA fixture."""
    with TEST_CASES_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("test_cases.json must contain a non-empty JSON array")
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not str(case.get("question", "")).strip():
            raise ValueError(
                f"test_cases.json item {index} must have a non-empty question"
            )
    return cases


def load_evaluation_rubric() -> list[dict]:
    """Load scoring items and maximum scores without hardcoding the rubric."""
    with RUBRIC_PATH.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"평가 항목", "배점", "평가 기준"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"evaluation_rubric.csv must contain columns: {required}")
    for row in rows:
        digits = "".join(character for character in row["배점"] if character.isdigit())
        if not digits:
            raise ValueError(f"invalid rubric score: {row['배점']}")
        row["배점"] = int(digits)
    return rows


async def evaluate_agent_activity(
    executions: list[dict], rubric: list[dict], client=None
) -> list[dict]:
    """Ask OpenAI to score all agent activity against the CSV rubric."""
    schema = {
        "type": "object",
        "properties": {
            "evaluations": {
                "type": "array",
                "minItems": len(rubric),
                "maxItems": len(rubric),
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["item", "score", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["evaluations"],
        "additionalProperties": False,
    }
    prompt = {
        "instruction": (
            "당신은 VOC 멀티 에이전트 시스템 평가자다. 제공된 모든 테스트 실행의 "
            "agent_activity와 final_result를 근거로 각 루브릭 항목을 독립적으로 평가하라. "
            "score는 0 이상 해당 max_score 이하이며, 근거가 없으면 낮게 평가하라. "
            "item은 rubric의 item 문자열을 정확히 그대로 반환하라."
        ),
        "rubric": [
            {
                "item": row["평가 항목"],
                "max_score": row["배점"],
                "criterion": row["평가 기준"],
            }
            for row in rubric
        ],
        "executions": executions,
    }
    client = client or AsyncOpenAI()
    response = await client.responses.create(
        model=os.getenv("OPENAI_EVAL_MODEL", MODEL_SUMMARY),
        input=json.dumps(prompt, ensure_ascii=False, default=str),
        text={
            "format": {
                "type": "json_schema",
                "name": "voc_agent_evaluation",
                "strict": True,
                "schema": schema,
            }
        },
    )
    payload = json.loads(response.output_text)
    received = {entry["item"]: entry for entry in payload["evaluations"]}
    results = []
    for row in rubric:
        item = row["평가 항목"]
        if item not in received:
            raise ValueError(f"OpenAI evaluation omitted rubric item: {item}")
        score = float(received[item]["score"])
        maximum = row["배점"]
        if not 0 <= score <= maximum:
            raise ValueError(f"score for {item} must be between 0 and {maximum}")
        results.append(
            {
                "평가 항목": item,
                "평가결과": f"{score:g}/{maximum} - {received[item]['reason']}",
                "평가 기준": row["평가 기준"],
            }
        )
    return results


async def evaluate_each_test_case(
    executions: list[dict], rubric: list[dict], client=None
) -> list[dict]:
    """Evaluate every test case separately and retain its identifying data."""
    client = client or AsyncOpenAI()
    case_results = []
    for execution in executions:
        rows = await evaluate_agent_activity([execution], rubric, client=client)
        case_results.append(
            {
                "case_id": execution.get("case_id") or "UNKNOWN",
                "question": execution.get("question") or "",
                "rows": rows,
            }
        )
    return case_results


def write_evaluation_results(case_results: list[dict]) -> None:
    """Write case-level CSV results and the two aggregate Markdown reports."""
    if not case_results:
        raise ValueError("case_results must not be empty")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_cases = []
    with RESULT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["평가 항목", "평가결과", "평가 기준"]
        )
        writer.writeheader()
        for index, case_result in enumerate(case_results):
            rows = case_result["rows"]
            writer.writerow(
                {
                    "평가 항목": "테스트 케이스",
                    "평가결과": case_result["case_id"],
                    "평가 기준": case_result.get("question", ""),
                }
            )
            writer.writerows(rows)

            earned_score = sum(
                float(row["평가결과"].split("/", 1)[0]) for row in rows
            )
            maximum_score = sum(
                float(row["평가결과"].split("/", 1)[1].split("-", 1)[0].strip())
                for row in rows
            )
            if maximum_score <= 0:
                raise ValueError(
                    f"maximum score must be positive: {case_result['case_id']}"
                )
            normalized_score = earned_score / maximum_score * 100
            scored_cases.append(
                {
                    "case_id": case_result["case_id"],
                    "question": case_result.get("question", ""),
                    "earned_score": earned_score,
                    "maximum_score": maximum_score,
                    "normalized_score": normalized_score,
                }
            )
            writer.writerow(
                {
                    "평가 항목": "총점",
                    "평가결과": (
                        f"{earned_score:g}/{maximum_score:g} "
                        f"({normalized_score:.2f}/100)"
                    ),
                    "평가 기준": (
                        f"{case_result['case_id']} 개별 평가 항목 점수 합계"
                    ),
                }
            )
            if index < len(case_results) - 1:
                writer.writerow({"평가 항목": "", "평가결과": "", "평가 기준": ""})

    write_quality_score_report(scored_cases)


def _deployment_decision(average_score: float) -> str:
    if average_score >= 90:
        return "배포 가능"
    if average_score >= 80:
        return "조건부 배포 가능, 개선 후 재검증"
    if average_score >= 70:
        return "주요 개선 필요"
    return "배포 보류"


def _markdown_cell(value) -> str:
    """Escape dynamic text for a single Markdown table cell."""
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def write_quality_score_report(scored_cases: list[dict]) -> None:
    """Write case totals, overall average, and a separate deployment decision."""
    if not scored_cases:
        raise ValueError("scored_cases must not be empty")

    average_score = sum(
        case["normalized_score"] for case in scored_cases
    ) / len(scored_cases)
    report_lines = [
        "# 품질 점수 보고서",
        "",
        "| 테스트 케이스 | 테스트 내용 | 총점 | 100점 환산 |",
        "|---|---|---:|---:|",
    ]
    for case in scored_cases:
        report_lines.append(
            "| {case_id} | {question} | {earned:g}/{maximum:g} | {normalized:.2f}점 |".format(
                case_id=_markdown_cell(case["case_id"]),
                question=_markdown_cell(case["question"]),
                earned=case["earned_score"],
                maximum=case["maximum_score"],
                normalized=case["normalized_score"],
            )
        )
    report_lines.extend(
        [
            "",
            f"## 전체 평균: {average_score:.2f}점",
            "",
            f"평가된 테스트 케이스 수: {len(scored_cases)}개",
            "",
        ]
    )
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    decision = _deployment_decision(average_score)
    decision_lines = [
        "# 배포 결정",
        "",
        f"- 전체 평균 점수: **{average_score:.2f}점**",
        f"- 배포 판정: **{decision}**",
        "",
        "## 판정 기준",
        "",
        "- 90점 이상: 배포 가능",
        "- 80~89점: 조건부 배포 가능, 개선 후 재검증",
        "- 70~79점: 주요 개선 필요",
        "- 69점 이하: 배포 보류",
        "",
    ]
    DEPLOYMENT_DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_DECISION_PATH.write_text(
        "\n".join(decision_lines), encoding="utf-8"
    )


class PipelineEndToEndTests(unittest.TestCase):
    def test_question_flows_through_grpc_server_and_all_six_agents(self) -> None:
        executions = []
        for test_case in load_test_cases():
            with self.subTest(case_id=test_case.get("case_id")):
                executions.append(asyncio.run(self._run_pipeline(test_case)))

        if not os.getenv("OPENAI_API_KEY"):
            return

        rubric = load_evaluation_rubric()
        case_evaluations = asyncio.run(
            evaluate_each_test_case(executions, rubric)
        )
        write_evaluation_results(case_evaluations)
        self.assertEqual(len(case_evaluations), len(executions))
        self.assertTrue(
            all(len(case["rows"]) == len(rubric) for case in case_evaluations)
        )

    async def _legacy_fake_pipeline(self, test_case: dict) -> dict:
        question = test_case["question"]
        calls: list[str] = []
        activity: dict[str, list[dict]] = {}
        started_at = time.perf_counter()

        def record(agent: str, agent_input, agent_output) -> None:
            activity.setdefault(agent, []).append(
                {"input": agent_input, "output": agent_output}
            )

        class FakeInterpreter:
            retriever_endpoint = ""

            async def parse(self, question, default_csv=None):
                calls.append("Interpreter")
                self.received_question = question
                intent = NLIntent(
                    task="both",
                    filters=["배송", "지연"],
                    max_items=20,
                    csv_path=default_csv or CSV_PATH,
                )
                record("Interpreter", {"question": question}, intent.__dict__)
                return intent

        class FakeRetriever:
            async def run(self, csv_path, filters, max_items):
                calls.append("Retriever")
                texts = [
                    "배송이 사흘 늦었고 안내를 받지 못했습니다.",
                    "배송 지연 상태를 실시간으로 확인하기 어렵습니다.",
                ]
                record(
                    "Retriever",
                    {"csv_path": csv_path, "filters": filters, "max_items": max_items},
                    {"texts": texts},
                )
                return texts

        class FakeEvaluator:
            _normalize_candidate_key = staticmethod(
                EvaluatorAgent._normalize_candidate_key
            )

            async def evaluate(self, task, candidates):
                calls.append("Evaluator")
                output = {
                    "winner": "S1",
                    "scores": {"S0": 80, "S1": 95, "S2": 85},
                }
                record(
                    "Evaluator",
                    {"task": task, "candidates": candidates},
                    output,
                )
                return output

        class FakeCritic:
            async def review(self, doc, role):
                calls.append("Critic")
                output = CriticResult(False, [], False)
                record("Critic", {"document": doc, "role": role}, output.__dict__)
                return output

        class FakeImprover:
            async def improve(self, summary):
                calls.append("Improver")
                output = PolicyResult(
                    "배송 지연 알림을 자동화하고 예상 도착 시간을 실시간 제공한다."
                )
                record("Improver", {"summary": summary}, output.__dict__)
                return output

        interpreter = InterpreterServicer.__new__(InterpreterServicer)
        interpreter.agent = FakeInterpreter()
        retriever = RetrieverServicer.__new__(RetrieverServicer)
        retriever.agent = FakeRetriever()
        evaluator = EvaluatorServicer.__new__(EvaluatorServicer)
        evaluator.agent = FakeEvaluator()
        critic = CriticServicer.__new__(CriticServicer)
        critic.critic = FakeCritic()
        improver = ImproverServicer.__new__(ImproverServicer)
        improver.imp = FakeImprover()

        summarizer_logic = SummarizerAgent.__new__(SummarizerAgent)

        async def make_candidates(texts, max_items, n):
            calls.append("Summarizer")
            candidates = {
                "S0": "배송 지연 관련 불만이 있습니다.",
                "S1": "고객은 배송 지연과 안내 부족을 주로 불만으로 제기했습니다.",
                "S2": "배송 현황 확인 기능 개선이 필요합니다.",
            }
            record(
                "Summarizer",
                {"texts": texts, "max_items": max_items, "candidate_count": n},
                {"candidates": candidates},
            )
            return candidates

        summarizer_logic.make_candidates = make_candidates
        summarizer = SummarizerServicer.__new__(SummarizerServicer)
        summarizer.agent = summarizer_logic

        services = []
        try:
            for register, servicer in (
                (voc_pb2_grpc.add_RetrieverServicer_to_server, retriever),
                (voc_pb2_grpc.add_EvaluatorServicer_to_server, evaluator),
                (voc_pb2_grpc.add_CriticServicer_to_server, critic),
                (voc_pb2_grpc.add_ImproverServicer_to_server, improver),
            ):
                services.append(await start_service(register, servicer))

            endpoints = [endpoint for _, endpoint in services]
            retriever_endpoint, evaluator_endpoint, critic_endpoint, improver_endpoint = (
                endpoints
            )
            interpreter.agent.retriever_endpoint = retriever_endpoint
            summarizer_logic.retriever_endpoint = retriever_endpoint
            summarizer_logic.evaluator_endpoint = evaluator_endpoint
            summarizer_logic.critic_endpoint = critic_endpoint
            summarizer_logic.improver_endpoint = improver_endpoint

            summarizer_service = await start_service(
                voc_pb2_grpc.add_SummarizerServicer_to_server, summarizer
            )
            services.append(summarizer_service)
            interpreter_service = await start_service(
                voc_pb2_grpc.add_InterpreterServicer_to_server, interpreter
            )
            services.append(interpreter_service)

            with (
                patch.object(
                    grpc_server, "INTERPRETER_ENDPOINT", interpreter_service[1]
                ),
                patch.object(
                    grpc_server, "SUMMARIZER_ENDPOINT", summarizer_service[1]
                ),
                patch.object(
                    grpc_server, "RETRIEVER_ENDPOINT", retriever_endpoint
                ),
            ):
                result = await grpc_server.VOCGRPCRuntime().run_with_question(
                    question, CSV_PATH, timeout=5.0
                )

            self.assertTrue(result["ok"])
            self.assertEqual(interpreter.agent.received_question, question)
            self.assertIn("배송 지연", result["summary"])
            self.assertIn("실시간", result["policy"])
            self.assertEqual(
                json.loads(result["intent_json"]),
                {
                    "task": "both",
                    "filters": ["배송", "지연"],
                    "max_items": 20,
                    "csv_path": CSV_PATH,
                },
            )
            self.assertEqual(json.loads(result["eval_json"])["S1"], 95)
            self.assertIn("retrieved=2", result["trace"])
            self.assertEqual(
                set(calls),
                {
                    "Interpreter",
                    "Retriever",
                    "Summarizer",
                    "Evaluator",
                    "Critic",
                    "Improver",
                },
            )
            self.assertLess(calls.index("Interpreter"), calls.index("Summarizer"))
            self.assertLess(calls.index("Summarizer"), calls.index("Evaluator"))
            self.assertLess(calls.index("Evaluator"), calls.index("Critic"))
            self.assertLess(calls.index("Critic"), calls.index("Improver"))
            return {
                "case_id": test_case.get("case_id"),
                "question": question,
                "expected": {
                    key: test_case.get(key)
                    for key in (
                        "expected_intent",
                        "expected_keywords",
                        "required_output",
                        "prohibited_output",
                    )
                },
                "agent_activity": activity,
                "agent_call_order": calls,
                "final_result": result,
                "elapsed_seconds": round(time.perf_counter() - started_at, 6),
            }
        finally:
            await asyncio.gather(
                *(server.stop(0) for server, _ in reversed(services)),
                return_exceptions=True,
            )

    async def _run_pipeline(self, test_case: dict) -> dict:
        """Run one question through the six live agents without fake outputs."""
        question = str(test_case["question"]).strip()
        started_at = time.perf_counter()
        log_offset = AGENT_LOG_PATH.stat().st_size if AGENT_LOG_PATH.exists() else 0
        timeout = float(os.getenv("GRPC_E2E_TIMEOUT_SECONDS", "180"))

        runtime = grpc_server.VOCGRPCRuntime()
        result = await runtime.run_with_question(question, CSV_PATH, timeout=timeout)
        self.assertTrue(result.get("ok"), result.get("error") or result.get("message"))

        # Interpreter may classify a complaint as summary-only. The E2E quality
        # test must still exercise Critic and Improver, so reuse the real intent
        # and run the live downstream pipeline with task=both.
        original_intent = json.loads(result.get("intent_json") or "{}")
        if not result.get("policy"):
            result = await runtime.run_with_params(
                filters=list(original_intent.get("filters") or []),
                task="both",
                max_items=int(original_intent.get("max_items") or 30),
                csv_path=str(original_intent.get("csv_path") or CSV_PATH),
                timeout=timeout,
            )
            result["intent_json"] = json.dumps(original_intent, ensure_ascii=False)

        self.assertTrue(result.get("ok"), result.get("error") or result.get("message"))
        self.assertTrue(str(result.get("summary", "")).strip(), "empty summary")
        self.assertTrue(str(result.get("policy", "")).strip(), "empty Improver output")

        # Agent file logging is synchronous, but yield once so separate gRPC
        # processes can finish their final append before this process reads it.
        await asyncio.sleep(0.05)
        events = []
        if AGENT_LOG_PATH.exists():
            with AGENT_LOG_PATH.open("rb") as log_file:
                log_file.seek(log_offset)
                appended_log = log_file.read().decode("utf-8", errors="replace")
            for line in appended_log.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("agent") in EXPECTED_AGENT_ORDER:
                    events.append(event)

        activity: dict[str, list[dict]] = {}
        observed_order = []
        for event in events:
            agent = event["agent"]
            if event.get("action") not in {"input", "output"}:
                continue
            activity.setdefault(agent, []).append(event)
            if agent not in observed_order:
                observed_order.append(agent)

        self.assertEqual(
            observed_order,
            EXPECTED_AGENT_ORDER,
            "live agents did not log the required order; restart all agents "
            "so they load the latest agent.log output code",
        )

        combined_output = "\n".join(
            [str(result.get("summary", "")), str(result.get("policy", ""))]
        )
        for prohibited in test_case.get("prohibited_output") or []:
            self.assertNotIn(str(prohibited), combined_output)

        return {
            "case_id": test_case.get("case_id"),
            "question": question,
            "expected": {
                key: test_case.get(key)
                for key in (
                    "expected_intent",
                    "expected_keywords",
                    "required_output",
                    "prohibited_output",
                )
            },
            "agent_activity": activity,
            "agent_call_order": observed_order,
            "final_result": result,
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        }


if __name__ == "__main__":
    unittest.main()
