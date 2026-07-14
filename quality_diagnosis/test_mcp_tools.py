"""Contract tests for the MCP tools exposed by :mod:`main`.

Pipeline calls are replaced by a recording runtime; MCP registration, argument
normalization, result propagation, fallback behavior, and health checks remain
real.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from utils import tools


class RecordingRuntime:
    def __init__(self):
        self.question_calls = []
        self.params_calls = []

    async def run_with_question(self, **kwargs):
        self.question_calls.append(kwargs)
        return {
            "ok": True,
            "summary": "배송 지연 관련 VOC 요약",
            "policy": "배송 상태 알림을 자동화한다.",
        }

    async def run_with_params(self, **kwargs):
        self.params_calls.append(kwargs)
        return {
            "ok": True,
            "summary": "필터 기반 VOC 요약",
            "policy": "상담 절차를 개선한다.",
        }


class FallbackRuntime(RecordingRuntime):
    async def run_with_question(self, **kwargs):
        self.question_calls.append(kwargs)
        raise RuntimeError("interpreter unavailable")


class MCPToolTests(unittest.TestCase):
    def test_main_exposes_the_three_expected_mcp_tools(self) -> None:
        self.assertIs(main.mcp, tools.mcp)

        registered = {tool.name for tool in asyncio.run(main.mcp.list_tools())}

        self.assertTrue(
            {"analyze_voc", "analyze_voc_nl_v2", "health_check"}.issubset(
                registered
            )
        )

    def test_analyze_voc_normalizes_arguments_and_calls_parameter_pipeline(self) -> None:
        runtime = RecordingRuntime()
        with patch.object(tools, "get_runtime", return_value=runtime):
            result = asyncio.run(
                tools.analyze_voc(
                    filters="배송, 지연",
                    task="both",
                    max_items=999,
                    csv_path="sample.csv",
                )
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"], "필터 기반 VOC 요약")
        self.assertEqual(len(runtime.params_calls), 1)
        call = runtime.params_calls[0]
        self.assertEqual(call["filters"], ["배송", "지연"])
        self.assertEqual(call["task"], "both")
        self.assertEqual(call["max_items"], 200)
        self.assertEqual(call["csv_path"], "sample.csv")
        self.assertEqual(call["timeout"], 180.0)

    def test_analyze_voc_nl_v2_forwards_the_natural_language_question(self) -> None:
        runtime = RecordingRuntime()
        question = "배송 지연 원인을 분석하고 정책을 제안해 줘"
        with patch.object(tools, "get_runtime", return_value=runtime):
            result = asyncio.run(
                tools.analyze_voc_nl_v2(question, csv_path="sample.csv")
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["policy"], "배송 상태 알림을 자동화한다.")
        self.assertEqual(
            runtime.question_calls,
            [
                {
                    "question": question,
                    "csv_path": "sample.csv",
                    "timeout": 180.0,
                }
            ],
        )

    def test_analyze_voc_nl_v2_uses_parameter_fallback_on_interpreter_error(self) -> None:
        runtime = FallbackRuntime()
        with patch.object(tools, "get_runtime", return_value=runtime):
            result = asyncio.run(
                tools.analyze_voc_nl_v2("배송 지연 분석", csv_path="sample.csv")
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["note"], "fallback: keyword-based run")
        self.assertEqual(len(runtime.question_calls), 1)
        self.assertEqual(len(runtime.params_calls), 1)
        fallback = runtime.params_calls[0]
        self.assertEqual(fallback["task"], "both")
        self.assertEqual(fallback["max_items"], 50)
        self.assertEqual(fallback["csv_path"], "sample.csv")
        self.assertTrue(fallback["filters"])

    def test_health_check_reports_existing_and_missing_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "voc.csv"
            csv_path.write_text("text\n배송 지연\n", encoding="utf-8")
            expected_size = csv_path.stat().st_size
            expected_path = csv_path.resolve()

            existing = asyncio.run(tools.health_check(str(csv_path)))
            missing = asyncio.run(
                tools.health_check(str(Path(directory) / "missing.csv"))
            )

        self.assertTrue(existing["ok"])
        self.assertEqual(existing["size"], expected_size)
        self.assertEqual(Path(existing["csv_path"]), expected_path)
        self.assertFalse(missing["ok"])
        self.assertIsNone(missing["size"])


if __name__ == "__main__":
    unittest.main()
