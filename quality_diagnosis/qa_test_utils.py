"""QA 테스트 데이터 형식 변환 도구."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).with_name("judge_cases.json")
DEFAULT_OUTPUT = Path(__file__).with_name("judge_cases.md")
DEFAULT_AGENT_LOG = Path(__file__).resolve().parents[1] / "agent.log"
DEFAULT_CASE_COUNT = 20


def build_judge_cases_from_agent_log(
    log_path: Path = DEFAULT_AGENT_LOG,
    output_path: Path = DEFAULT_INPUT,
    case_count: int = DEFAULT_CASE_COUNT,
) -> list[dict[str, Any]]:
    """agent.log의 순차 이벤트를 question/candidates/policy 케이스로 매핑한다.

    각 Interpreter input부터 다음 Interpreter input 직전까지 처음 나타나는
    Summarizer make_candidates와 Improver improve를 한 케이스로 묶는다.
    둘 중 나타나지 않은 값은 JSON null로 기록한다. 동일 policy가 다시 기록되는
    ``rpc: Improve`` 이벤트는 ``operation: improve``가 아니므로 자동 제외된다.
    """
    if case_count <= 0:
        raise ValueError("case_count는 1 이상이어야 합니다.")

    cases: list[dict[str, Any]] = []
    question: str | None = None
    candidates: dict[str, Any] | None = None
    policy: str | None = None

    def finalize_current_case() -> bool:
        """현재 질문을 누락 필드는 None(JSON null)으로 채워 확정한다."""
        nonlocal question, candidates, policy
        if question is None or len(cases) >= case_count:
            return len(cases) >= case_count
        cases.append(
            {
                "question": question,
                "candidates": candidates,
                "policy": policy,
            }
        )
        question = None
        candidates = None
        policy = None
        return len(cases) >= case_count

    with log_path.open(encoding="utf-8-sig") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"agent.log {line_number}행이 올바른 JSON이 아닙니다."
                ) from exc

            if (
                event.get("agent") == "Interpreter"
                and event.get("action") == "input"
                and event.get("question")
            ):
                if finalize_current_case():
                    break
                question = str(event["question"])
                candidates = None
                policy = None
                continue

            if (
                question is not None
                and candidates is None
                and event.get("agent") == "Summarizer"
                and event.get("action") == "output"
                and event.get("operation") == "make_candidates"
                and isinstance(event.get("candidates"), dict)
            ):
                candidates = event["candidates"]
                continue

            if (
                question is not None
                and policy is None
                and event.get("agent") == "Improver"
                and event.get("action") == "output"
                and event.get("operation") == "improve"
                and event.get("policy")
            ):
                policy = str(event["policy"])

    if len(cases) < case_count:
        finalize_current_case()

    if len(cases) != case_count:
        raise ValueError(
            f"agent.log에서 {case_count}개 케이스를 요청했지만 "
            f"{len(cases)}개만 완성되었습니다."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cases


def _shift_policy_headings(policy: str) -> str:
    """정책 내부 제목을 케이스의 하위 제목으로 배치한다."""

    def replace_heading(match: re.Match[str]) -> str:
        level = min(len(match.group(1)) + 3, 6)
        return f"{'#' * level} {match.group(2)}"

    return re.sub(r"(?m)^(#{1,6})\s+(.+)$", replace_heading, policy.strip())


def cases_to_markdown(cases: list[dict[str, Any]]) -> str:
    """judge case 목록을 사람이 읽기 쉬운 Markdown 문자열로 변환한다."""
    lines = ["# LLM Judge Cases", "", f"총 케이스 수: {len(cases)}개", ""]

    for index, case in enumerate(cases, start=1):
        missing = {"question", "candidates", "policy"} - case.keys()
        if missing:
            raise ValueError(f"{index}번 케이스에 항목이 없습니다: {sorted(missing)}")
        if case["candidates"] is not None and not isinstance(case["candidates"], dict):
            raise ValueError(f"{index}번 케이스의 candidates는 객체여야 합니다.")

        lines.extend(
            [
                f"## Case {index}",
                "",
                "### Question",
                "",
                str(case["question"]).strip(),
                "",
                "### Candidates",
                "",
            ]
        )
        if case["candidates"] is None:
            lines.append("NULL")
        else:
            for name, candidate in case["candidates"].items():
                lines.append(f"- **{name}**: {str(candidate).strip()}")

        policy = (
            "NULL"
            if case["policy"] is None
            else _shift_policy_headings(str(case["policy"]))
        )
        lines.extend(["", "### Policy", "", policy, "", "---", ""])

    return "\n".join(lines).rstrip() + "\n"


def convert_judge_cases_json_to_md(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> int:
    """judge_cases.json을 judge_cases.md로 변환하고 케이스 수를 반환한다."""
    with input_path.open(encoding="utf-8-sig") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("judge_cases.json의 최상위 값은 배열이어야 합니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cases_to_markdown(cases), encoding="utf-8")
    return len(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "agent.log에서 judge_cases.json을 만들고 Markdown 문서로 변환합니다."
        )
    )
    parser.add_argument("--agent-log", type=Path, default=DEFAULT_AGENT_LOG)
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_judge_cases_from_agent_log(
        args.agent_log,
        args.input,
        args.case_count,
    )
    print(f"JSON 생성 완료: {args.input} ({len(cases)}개 케이스)")
    count = convert_judge_cases_json_to_md(args.input, args.output)
    print(f"변환 완료: {args.output} ({count}개 케이스)")


if __name__ == "__main__":
    main()
