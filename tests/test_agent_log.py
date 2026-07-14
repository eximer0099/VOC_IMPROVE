import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from utils.agent_log import agent_file_event, log_response_time


class AgentResponseTimeLogTests(unittest.TestCase):
    def test_logs_rpc_response_time_to_stderr(self):
        @log_response_time("TestAgent")
        async def respond():
            return "ok"

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = asyncio.run(respond())

        self.assertEqual(result, "ok")
        line = stderr.getvalue().strip()
        self.assertTrue(line.startswith("[VOC_AGENT] "))
        event = json.loads(line.removeprefix("[VOC_AGENT] "))
        self.assertEqual(event["agent"], "TestAgent")
        self.assertEqual(event["action"], "response_time")
        self.assertEqual(event["rpc"], "respond")
        self.assertGreaterEqual(event["elapsed_ms"], 0)

    def test_logs_response_time_when_rpc_fails(self):
        @log_response_time("TestAgent")
        async def fail():
            raise RuntimeError("failure")

        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(RuntimeError):
            asyncio.run(fail())

        event = json.loads(
            stderr.getvalue().strip().removeprefix("[VOC_AGENT] ")
        )
        self.assertEqual(event["rpc"], "fail")
        self.assertIn("elapsed_ms", event)

    def test_logs_waiting_message_when_response_is_delayed(self):
        @log_response_time("SlowAgent")
        async def respond_slowly():
            await asyncio.sleep(0.03)
            return "ok"

        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {"AGENT_DELAY_WARNING_SECONDS": "0.01"}),
            redirect_stderr(stderr),
        ):
            result = asyncio.run(respond_slowly())

        self.assertEqual(result, "ok")
        events = [
            json.loads(line.removeprefix("[VOC_AGENT] "))
            for line in stderr.getvalue().splitlines()
        ]
        delayed = next(event for event in events if event["action"] == "response_delayed")
        self.assertEqual(delayed["agent"], "SlowAgent")
        self.assertEqual(delayed["rpc"], "respond_slowly")
        self.assertEqual(delayed["status"], "waiting")
        self.assertIn("지연", delayed["message"])
        self.assertIn("기다려", delayed["message"])
        self.assertEqual(events[-1]["action"], "response_time")

    def test_writes_interpreter_input_and_improver_output_to_agent_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "agent.log"
            with patch.dict(os.environ, {"AGENT_LOG_PATH": str(log_path)}):
                agent_file_event(
                    "Interpreter", "input", question="결제 내역이 보이지 않습니다."
                )
                agent_file_event(
                    "Improver", "output", policy="결제-주문 동기화를 점검한다."
                )

            events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(events[0]["agent"], "Interpreter")
        self.assertEqual(events[0]["action"], "input")
        self.assertIn("결제 내역", events[0]["question"])
        self.assertEqual(events[1]["agent"], "Improver")
        self.assertEqual(events[1]["action"], "output")
        self.assertIn("동기화", events[1]["policy"])


if __name__ == "__main__":
    unittest.main()
