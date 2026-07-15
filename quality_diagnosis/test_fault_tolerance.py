"""Fault-tolerance checks for malformed or missing pipeline inputs.

The cases avoid external LLM and network calls while verifying that invalid
input produces defaults, explicit exceptions, or controlled gRPC aborts.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import AsyncMock, patch

import grpc

import grpc_server
import voc_pb2
from agents.critic import CriticResult, CriticServicer
from agents.improver import ImproverServicer, PolicyResult
from agents.interpreter import NLInterpreterAgent
from agents.retriever import RetrieverAgent, RetrieverServicer
from agents.summarizer import SummarizerAgent


class EmptyLLMResponse:
    """Responses API double whose output is intentionally malformed."""

    class Responses:
        async def create(self, **kwargs):
            return object()

    def __init__(self):
        self.responses = self.Responses()


class ControlledAbort(RuntimeError):
    """Raised by FakeContext to prove an RPC used context.abort safely."""


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = ""

    async def abort(self, code, details):
        self.code = code
        self.details = details
        raise ControlledAbort(details)


class FaultToleranceTests(unittest.TestCase):
    def test_no_related_voc_is_reported_clearly_to_stderr(self) -> None:
        agent = RetrieverAgent()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "voc.csv"
            csv_path.write_text("고객ID,불만내용\nCUST001,배송 지연\n", encoding="utf-8")
            with (
                redirect_stderr(stderr),
                patch.object(
                    agent,
                    "_append_missing_input",
                    new=AsyncMock(return_value=""),
                ),
            ):
                result = asyncio.run(
                    agent.run(
                        str(csv_path),
                        filters=["__NO_MATCH_FAULT_TEST_7F3A91C2__"],
                        max_items=30,
                    )
                )

        self.assertEqual(result, [])
        events = [
            json.loads(line.removeprefix("[VOC_AGENT] "))
            for line in stderr.getvalue().splitlines()
        ]
        event = next(item for item in events if item["action"] == "no_related_data")
        self.assertEqual(event["agent"], "Retriever")
        self.assertEqual(event["retrieved_count"], 0)
        self.assertIn("관련 데이터 없음", event["message"])

    def test_duplicate_server_port_reports_port_in_use(self) -> None:
        class OccupiedPortServer:
            def add_insecure_port(self, endpoint):
                raise RuntimeError("Failed to bind to address")

        with self.assertRaisesRegex(RuntimeError, "포트 사용 중") as caught:
            grpc_server.bind_agent_port(
                OccupiedPortServer(), "127.0.0.1:6002", "Retriever"
            )

        self.assertIn("127.0.0.1:6002", str(caught.exception))
        self.assertIn("기존 서버를 종료", str(caught.exception))

    def test_stopped_retriever_returns_clear_search_unavailable_error(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            unused_endpoint = f"127.0.0.1:{probe.getsockname()[1]}"

        with patch.object(grpc_server, "RETRIEVER_ENDPOINT", unused_endpoint):
            result = asyncio.run(
                grpc_server.VOCGRPCRuntime().run_with_params(
                    filters=["배송"],
                    task="both",
                    max_items=30,
                    csv_path="voc.csv",
                    timeout=0.1,
                )
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["trace"], "retriever_unavailable")
        self.assertIn("Retriever 검색 불가", result["error"])
        self.assertIn("Retriever 터미널", result["message"])

    def test_empty_question_and_malformed_llm_response_use_safe_defaults(self) -> None:
        agent = NLInterpreterAgent.__new__(NLInterpreterAgent)
        agent.client = EmptyLLMResponse()

        intent = asyncio.run(agent.parse("", default_csv="fallback.csv"))

        self.assertEqual(intent.task, "both")
        self.assertEqual(intent.filters, [])
        self.assertEqual(intent.max_items, 30)
        self.assertEqual(intent.csv_path, "fallback.csv")

    def test_empty_voc_input_returns_a_nonempty_summary_candidate(self) -> None:
        agent = SummarizerAgent.__new__(SummarizerAgent)

        candidates = asyncio.run(agent.make_candidates([], max_items=30, n=3))

        self.assertEqual(list(candidates), ["S0"])
        self.assertTrue(candidates["S0"].strip())

    def test_missing_csv_is_reported_explicitly(self) -> None:
        missing = Path(tempfile.gettempdir()) / "voc-file-that-does-not-exist.csv"
        agent = RetrieverAgent()

        with self.assertRaisesRegex(FileNotFoundError, "VOC 데이터 파일 오류"):
            asyncio.run(agent.run(str(missing), filters=[], max_items=30))

    def test_missing_csv_rpc_returns_not_found_with_data_file_guidance(self) -> None:
        servicer = RetrieverServicer.__new__(RetrieverServicer)
        servicer.agent = RetrieverAgent()
        context = FakeContext()
        request = voc_pb2.RetrieveReq(
            csv_path="missing-voc.csv", filters=[], max_items=30
        )

        with self.assertRaises(ControlledAbort):
            asyncio.run(servicer.Retrieve(request, context))

        self.assertEqual(context.code, grpc.StatusCode.NOT_FOUND)
        self.assertIn("VOC 데이터 파일 오류", context.details)
        self.assertIn("A2A_VOC_CSV", context.details)

    def test_retriever_rpc_converts_internal_failure_to_controlled_grpc_abort(self) -> None:
        class FailingRetriever:
            async def run(self, csv_path, filters, max_items):
                raise ValueError("invalid retrieval input")

        servicer = RetrieverServicer.__new__(RetrieverServicer)
        servicer.agent = FailingRetriever()
        context = FakeContext()
        request = voc_pb2.RetrieveReq(csv_path="", filters=[], max_items=-1)

        with self.assertRaises(ControlledAbort):
            asyncio.run(servicer.Retrieve(request, context))

        self.assertEqual(context.code, grpc.StatusCode.INTERNAL)
        self.assertIn("Retriever error", context.details)
        self.assertNotIn("Traceback", context.details)

    def test_empty_critic_role_falls_back_to_summary(self) -> None:
        class RecordingCritic:
            role = None

            async def review(self, doc, role):
                self.role = role
                return CriticResult(False, [], False)

        critic = RecordingCritic()
        servicer = CriticServicer.__new__(CriticServicer)
        servicer.critic = critic
        response = asyncio.run(
            servicer.Review(voc_pb2.ReviewReq(doc="요약", role=""), FakeContext())
        )

        self.assertEqual(critic.role, "summary")
        self.assertEqual(response.summary, "요약")

    def test_empty_policy_result_is_replaced_with_safe_fallback(self) -> None:
        class EmptyImprover:
            async def improve(self, summary):
                return PolicyResult("")

        servicer = ImproverServicer.__new__(ImproverServicer)
        servicer.imp = EmptyImprover()
        response = asyncio.run(
            servicer.Improve(voc_pb2.PolicyReq(summary=""), FakeContext())
        )

        self.assertTrue(response.policy.strip())
        self.assertGreaterEqual(len(response.policy), 20)


if __name__ == "__main__":
    unittest.main()
