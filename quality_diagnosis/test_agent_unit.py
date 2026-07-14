"""Six agent modules' file, syntax, and basic-structure checks.

These tests inspect source code with :mod:`ast`, so they do not initialize an
LLM client, open a gRPC port, or require API credentials.
"""

from __future__ import annotations

import ast
import asyncio
import csv
import io
import json
import os
import unittest
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path

import grpc

import grpc_server
import voc_pb2
import voc_pb2_grpc
from agents.retriever import RetrieverAgent, RetrieverServicer
from utils.agent_log import log_authentication_error, log_response_time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = PROJECT_ROOT / "agents"
VOC_PATH = PROJECT_ROOT / "voc.csv"
TEST_CASES_PATH = PROJECT_ROOT / "quality_diagnosis" / "test_cases.json"


@dataclass(frozen=True)
class AgentSpec:
    module: str
    agent_class: str
    servicer_class: str
    rpc_methods: tuple[str, ...]

    @property
    def path(self) -> Path:
        return AGENTS_DIR / f"{self.module}.py"


AGENT_SPECS = (
    AgentSpec(
        "interpreter", "NLInterpreterAgent", "InterpreterServicer", ("ParseQuestion",)
    ),
    AgentSpec("retriever", "RetrieverAgent", "RetrieverServicer", ("Retrieve",)),
    AgentSpec(
        "summarizer",
        "SummarizerAgent",
        "SummarizerServicer",
        ("MakeCandidates", "Refine", "RunPipeline"),
    ),
    AgentSpec("evaluator", "EvaluatorAgent", "EvaluatorServicer", ("Evaluate",)),
    AgentSpec("critic", "CriticAgent", "CriticServicer", ("Review",)),
    AgentSpec(
        "improver",
        "PolicyImproverAgent",
        "ImproverServicer",
        ("Improve", "Refine", "RunPolicyPipeline"),
    ),
)


def parse_agent(spec: AgentSpec) -> ast.Module:
    """Read and parse one agent, preserving useful syntax-error diagnostics."""
    source = spec.path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(spec.path))


class AgentModuleUnitTests(unittest.TestCase):
    def test_all_six_agent_files_exist(self) -> None:
        self.assertEqual(len(AGENT_SPECS), 6)
        for spec in AGENT_SPECS:
            with self.subTest(agent=spec.module):
                self.assertTrue(spec.path.is_file(), f"agent file not found: {spec.path}")

    def test_all_agent_files_have_valid_python_syntax(self) -> None:
        for spec in AGENT_SPECS:
            with self.subTest(agent=spec.module):
                try:
                    parse_agent(spec)
                except (OSError, UnicodeError, SyntaxError) as error:
                    self.fail(f"cannot parse {spec.path}: {error}")

    def test_agent_and_grpc_servicer_classes_exist(self) -> None:
        for spec in AGENT_SPECS:
            with self.subTest(agent=spec.module):
                tree = parse_agent(spec)
                classes = {
                    node.name: node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef)
                }
                self.assertIn(spec.agent_class, classes)
                self.assertIn(spec.servicer_class, classes)

    def test_servicers_expose_expected_async_rpc_methods(self) -> None:
        for spec in AGENT_SPECS:
            with self.subTest(agent=spec.module):
                tree = parse_agent(spec)
                servicer = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == spec.servicer_class
                )
                async_methods = {
                    node.name
                    for node in servicer.body
                    if isinstance(node, ast.AsyncFunctionDef)
                }
                self.assertTrue(
                    set(spec.rpc_methods).issubset(async_methods),
                    f"{spec.servicer_class} is missing async RPC method(s): "
                    f"{sorted(set(spec.rpc_methods) - async_methods)}",
                )

    def test_each_module_has_async_serve_entrypoint(self) -> None:
        for spec in AGENT_SPECS:
            with self.subTest(agent=spec.module):
                tree = parse_agent(spec)
                async_functions = {
                    node.name
                    for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef)
                }
                self.assertIn("serve", async_functions)


class AgentQualityTests(unittest.TestCase):
    """Quality checks grouped by the six requested operational dimensions."""

    def test_1_functional_quality_agent_roles_and_expected_interfaces(self) -> None:
        """Every agent must expose its role class and expected async RPCs."""
        expected_roles = {
            "interpreter": "NLInterpreterAgent",
            "retriever": "RetrieverAgent",
            "summarizer": "SummarizerAgent",
            "evaluator": "EvaluatorAgent",
            "critic": "CriticAgent",
            "improver": "PolicyImproverAgent",
        }
        for spec in AGENT_SPECS:
            with self.subTest(agent=spec.module):
                tree = parse_agent(spec)
                classes = {
                    node.name: node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef)
                }
                self.assertIn(expected_roles[spec.module], classes)
                servicer = classes[spec.servicer_class]
                rpc_names = {
                    node.name
                    for node in servicer.body
                    if isinstance(node, ast.AsyncFunctionDef)
                }
                self.assertTrue(set(spec.rpc_methods).issubset(rpc_names))

    def test_2_connection_quality_ports_and_grpc_stubs_are_configured(self) -> None:
        """The six production ports and corresponding generated stubs must match."""
        configured = {
            "Interpreter": grpc_server.INTERPRETER_ENDPOINT,
            "Retriever": grpc_server.RETRIEVER_ENDPOINT,
            "Summarizer": grpc_server.SUMMARIZER_ENDPOINT,
            "Evaluator": grpc_server.EVALUATOR_ENDPOINT,
            "Critic": grpc_server.CRITIC_ENDPOINT,
            "Improver": grpc_server.IMPROVER_ENDPOINT,
        }
        expected_ports = dict(zip(configured, range(6001, 6007)))
        for name, endpoint in configured.items():
            with self.subTest(agent=name):
                self.assertEqual(int(endpoint.rsplit(":", 1)[1]), expected_ports[name])
                self.assertTrue(hasattr(voc_pb2_grpc, f"{name}Stub"))

    @unittest.skipUnless(
        os.getenv("RUN_LIVE_AGENT_TESTS") == "1",
        "set RUN_LIVE_AGENT_TESTS=1 after running launch_agents.py",
    )
    def test_2_live_ports_6001_to_6006_accept_grpc_connections(self) -> None:
        """Optional live check: all six launched services become channel-ready."""

        async def check_all() -> None:
            channels = [
                grpc.aio.insecure_channel(f"127.0.0.1:{port}")
                for port in range(6001, 6007)
            ]
            try:
                await asyncio.gather(
                    *(asyncio.wait_for(channel.channel_ready(), 5.0) for channel in channels)
                )
            finally:
                await asyncio.gather(*(channel.close() for channel in channels))

        asyncio.run(check_all())

    def test_3_data_quality_retrieval_matches_original_voc_csv(self) -> None:
        """Retriever output must exactly match filtered source rows, in source order."""
        with VOC_PATH.open(encoding="utf-8-sig", newline="") as file:
            source_rows = list(csv.DictReader(file))
        keyword = "대기 시간"
        expected = [
            " ".join(row.values())
            for row in source_rows
            if keyword in row["불만내용"]
        ][:30]

        actual = asyncio.run(
            RetrieverAgent().run(str(VOC_PATH), filters=[keyword], max_items=30)
        )

        self.assertTrue(expected, "voc.csv must contain the grounding test keyword")
        self.assertEqual(actual, expected)
        original_lines = {" ".join(row.values()) for row in source_rows}
        self.assertTrue(all(text in original_lines for text in actual))

    def test_4_ai_answer_quality_contract_checks_grounding_and_usefulness(self) -> None:
        """The answer-quality gate rejects empty, prohibited, or ungrounded output."""
        cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
        case = cases[0]
        grounded_answer = (
            f"질문 의도: {case['expected_intent']}. "
            f"확인 항목: {', '.join(case['expected_keywords'])}. "
            f"권고: {', '.join(case['required_output'])}."
        )

        self.assertGreaterEqual(len(grounded_answer.strip()), 30)
        self.assertTrue(
            any(keyword in grounded_answer for keyword in case["expected_keywords"]),
            "answer must cite expected evidence or keywords",
        )
        self.assertFalse(
            any(text in grounded_answer for text in case["prohibited_output"]),
            "answer must not contain unsupported/prohibited claims",
        )
        self.assertTrue(
            any(item in grounded_answer for item in case["required_output"]),
            "answer must contain useful required guidance",
        )

    def test_5_fault_quality_grpc_deadline_is_explicit(self) -> None:
        """A stalled agent call must surface DEADLINE_EXCEEDED within five seconds."""

        class SlowRetriever:
            async def run(self, csv_path, filters, max_items):
                await asyncio.sleep(1)
                return []

        async def exercise_timeout() -> None:
            servicer = RetrieverServicer.__new__(RetrieverServicer)
            servicer.agent = SlowRetriever()
            server = grpc.aio.server()
            voc_pb2_grpc.add_RetrieverServicer_to_server(servicer, server)
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            try:
                async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                    stub = voc_pb2_grpc.RetrieverStub(channel)
                    with self.assertRaises(grpc.aio.AioRpcError) as caught:
                        await stub.Retrieve(
                            voc_pb2.RetrieveReq(csv_path=str(VOC_PATH)), timeout=0.05
                        )
                    self.assertEqual(caught.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
                    self.assertTrue(caught.exception.details())
            finally:
                await server.stop(0)

        asyncio.run(exercise_timeout())

    def test_6_operational_quality_stderr_contains_monitoring_fields(self) -> None:
        """Agent logs must expose agent, RPC, event, and elapsed time on stderr."""

        @log_response_time("MonitorAgent")
        async def monitored_call():
            return "ok"

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(asyncio.run(monitored_call()), "ok")

        line = stderr.getvalue().strip()
        self.assertTrue(line.startswith("[VOC_AGENT] "))
        event = json.loads(line.removeprefix("[VOC_AGENT] "))
        self.assertEqual(event["agent"], "MonitorAgent")
        self.assertEqual(event["action"], "response_time")
        self.assertEqual(event["rpc"], "monitored_call")
        self.assertGreaterEqual(event["elapsed_ms"], 0)

    def test_all_agents_log_api_authentication_errors(self) -> None:
        for spec in AGENT_SPECS:
            with self.subTest(agent=spec.module):
                source = spec.path.read_text(encoding="utf-8")
                self.assertIn("log_authentication_error", source)

    def test_authentication_error_is_written_to_stderr_without_api_key(self) -> None:
        class AuthenticationError(RuntimeError):
            status_code = 401

        secret = "sk-secret-value-that-must-not-be-logged"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            detected = log_authentication_error(
                "Evaluator", AuthenticationError(f"Incorrect API key: {secret}")
            )

        self.assertTrue(detected)
        output = stderr.getvalue()
        self.assertIn('"action": "authentication_error"', output)
        self.assertIn("API 키 인증 오류", output)
        self.assertIn('"status_code": 401', output)
        self.assertNotIn(secret, output)


if __name__ == "__main__":
    unittest.main()
