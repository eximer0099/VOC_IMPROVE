# ================================================
# File: grpc_server.py
# Role: A2A VOC Orchestrator (gRPC 기반 클라이언트)
# ================================================

# ============ 표준 라이브러리 및 타입 힌트 ============
# Python 3.7+ 호환성을 위한 annotations 가져오기 (타입 힌트 지연 평가)
from __future__ import annotations
import asyncio
# 운영체제 관련 기능 (환경변수 읽기 등)
import os
# JSON 데이터 직렬화/역직렬화
import json
import re
# 타입 힌트를 위한 타입 정의들
from typing import Dict, Any, Optional, List
# gRPC 라이브러리 (비동기 클라이언트/서버 통신)
import grpc
# Protocol Buffers로 생성된 메시지 및 서비스 정의
import voc_pb2
import voc_pb2_grpc

# ============ 프로젝트 내부 모듈 임포트 ============
# settings.py에서 기본 CSV 경로를 불러오는 방식으로 통일
# 이렇게 하면 CSV 경로 설정이 한 곳에서 관리됩니다
from utils.settings import DEFAULT_CSV
from utils.agent_log import agent_event

# ============ gRPC 에이전트 엔드포인트 설정 ============
# 각 에이전트 서비스의 네트워크 주소를 환경변수에서 읽어옵니다
# 환경변수가 없으면 기본값(localhost)을 사용합니다
# 각 에이전트는 독립적인 포트에서 실행됩니다
INTERPRETER_ENDPOINT = os.environ.get("INTERPRETER_ENDPOINT", "localhost:6001")  # 자연어 질의 해석 서비스
RETRIEVER_ENDPOINT   = os.environ.get("RETRIEVER_ENDPOINT",   "localhost:6002")  # VOC 데이터 검색 서비스
SUMMARIZER_ENDPOINT  = os.environ.get("SUMMARIZER_ENDPOINT",  "localhost:6003")  # 요약 생성 서비스
EVALUATOR_ENDPOINT   = os.environ.get("EVALUATOR_ENDPOINT",   "localhost:6004")  # 요약 평가 서비스
CRITIC_ENDPOINT      = os.environ.get("CRITIC_ENDPOINT",      "localhost:6005")  # 요약/정책 비평 서비스
IMPROVER_ENDPOINT    = os.environ.get("IMPROVER_ENDPOINT",    "localhost:6006")  # 정책 개선안 생성 서비스


def bind_agent_port(server, endpoint: str, agent: str) -> int:
    """Bind a gRPC server and clearly report an occupied port."""
    try:
        bound_port = server.add_insecure_port(endpoint)
    except RuntimeError as error:
        message = (
            f"{agent} 서버 시작 실패: 포트 사용 중 ({endpoint}). "
            "동일 포트의 기존 서버를 종료한 후 다시 실행하세요."
        )
        agent_event(agent, "port_in_use", endpoint=endpoint, error=str(error))
        raise RuntimeError(message) from error

    if bound_port == 0:
        message = (
            f"{agent} 서버 시작 실패: 포트 사용 중 ({endpoint}). "
            "동일 포트의 기존 서버를 종료한 후 다시 실행하세요."
        )
        agent_event(agent, "port_in_use", endpoint=endpoint)
        raise RuntimeError(message)
    return bound_port


class VOCGRPCRuntime:
    """
    A2A VOC 전체 파이프라인 실행기
    MCP 서버에서 호출되는 인터페이스
    """

    # ============ 초기화 메서드 ============
    def __init__(self):
        # 각 에이전트는 모듈 상단의 INTERPRETER_ENDPOINT, SUMMARIZER_ENDPOINT 등 환경 변수 기반 상수를 사용합니다
        pass

    @staticmethod
    def _normalize_topic_text(value: str) -> str:
        """Normalize text for a deterministic intent/topic comparison."""
        return re.sub(r"[^0-9a-zA-Z가-힣]+", " ", str(value).lower()).strip()

    @classmethod
    def validate_intent_topic(
        cls, intent: Dict[str, Any], summary: str, policy: str
    ) -> Dict[str, Any]:
        """Check that the final response still covers the Interpreter intent."""
        task = str(intent.get("task") or "both").lower()
        normalized_summary = cls._normalize_topic_text(summary)
        normalized_policy = cls._normalize_topic_text(policy)
        final_text = f"{normalized_summary} {normalized_policy}".strip()

        filters = [
            cls._normalize_topic_text(item)
            for item in (intent.get("filters") or [])
            if cls._normalize_topic_text(item)
        ]
        topic_terms = []
        for value in filters:
            topic_terms.append(value)
            topic_terms.extend(token for token in value.split() if len(token) >= 2)
        topic_terms = list(dict.fromkeys(topic_terms))
        matched_terms = [term for term in topic_terms if term in final_text]

        missing_outputs = []
        if task in {"summary", "both"} and not normalized_summary:
            missing_outputs.append("summary")
        if task in {"policy", "both"} and not normalized_policy:
            missing_outputs.append("policy")

        reasons = []
        if not topic_terms:
            reasons.append("intent에 검증 가능한 주제어가 없습니다.")
        elif not matched_terms:
            reasons.append("최종 응답에서 intent 주제어를 찾을 수 없습니다.")
        if missing_outputs:
            reasons.append(
                "요청 task에 필요한 출력이 없습니다: " + ", ".join(missing_outputs)
            )

        return {
            "passed": not reasons,
            "task": task,
            "topic_terms": topic_terms,
            "matched_terms": matched_terms,
            "missing_outputs": missing_outputs,
            "reasons": reasons,
        }

    @classmethod
    def _apply_intent_topic_guardrail(
        cls, intent: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Block a successful response when its topic diverges from intent."""
        guardrail = cls.validate_intent_topic(
            intent, result.get("summary", ""), result.get("policy", "")
        )
        result["intent_guardrail_json"] = json.dumps(guardrail, ensure_ascii=False)
        if not result.get("ok") or guardrail["passed"]:
            return result

        message = "Intent 주제 일치 가드레일 실패: " + " ".join(
            guardrail["reasons"]
        )
        agent_event("Orchestrator", "intent_topic_guardrail_failed", **guardrail)
        result.update(
            {
                "ok": False,
                "summary": "",
                "policy": "",
                "message": message,
                "error": message,
                "trace": "; ".join(
                    part
                    for part in (
                        result.get("trace", ""),
                        "intent_topic_guardrail_failed",
                    )
                    if part
                ),
            }
        )
        return result

    async def _check_retriever(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Return a clear failure result when the Retriever service is down."""
        channel = grpc.aio.insecure_channel(RETRIEVER_ENDPOINT)
        try:
            await asyncio.wait_for(
                channel.channel_ready(), timeout=max(0.1, min(timeout, 5.0))
            )
            return None
        except (TimeoutError, grpc.RpcError, OSError) as error:
            message = (
                f"Retriever 검색 불가: Retriever 서비스({RETRIEVER_ENDPOINT})에 "
                "연결할 수 없습니다. Retriever 터미널이 실행 중인지 확인하세요."
            )
            agent_event(
                "Orchestrator",
                "retriever_unavailable",
                endpoint=RETRIEVER_ENDPOINT,
                error=str(error),
            )
            return {
                "ok": False,
                "summary": "",
                "policy": "",
                "intent_json": "{}",
                "eval_json": "{}",
                "summary_critic_json": "{}",
                "trace": "retriever_unavailable",
                "message": message,
                "error": message,
            }
        finally:
            await channel.close()

    # ============ 자연어 기반 실행 메서드 ============
    # 사용자의 자연어 질의를 받아서 전체 VOC 분석 파이프라인을 실행합니다
    # 이 메서드는 Interpreter 에이전트를 먼저 호출하여 질의를 구조화된 파라미터로 변환합니다
    async def run_with_question(
        self,
        question: str,
        csv_path: Optional[str],
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        """
        자연어 질의를 받아 VOC 분석 파이프라인을 실행합니다.
        
        Args:
            question: 사용자의 자연어 질의 (예: "상담 대기 시간 관련 불만 분석")
            csv_path: VOC 데이터 CSV 파일 경로 (None이면 기본값 사용)
            timeout: 각 gRPC 호출의 타임아웃 시간(초)
            
        Returns:
            Dict: 분석 결과 (summary, policy, trace 등 포함)
        """

        # CSV 경로 우선순위 결정:
        #   1순위: 사용자가 명시적으로 제공한 csv_path
        #   2순위: settings.py의 DEFAULT_CSV
        final_csv = csv_path or DEFAULT_CSV
        agent_event("Orchestrator", "pipeline_started", mode="question",
                    question=question, csv_path=final_csv)

        retriever_error = await self._check_retriever(timeout)
        if retriever_error:
            return retriever_error

        # ============ 1단계: Interpreter 에이전트 호출 ============
        # 자연어 질의를 구조화된 파라미터(task, filters, max_items 등)로 변환합니다
        # insecure_channel은 TLS 없이 통신합니다 (로컬 개발 환경용)
        agent_event("Orchestrator", "call_interpreter")
        async with grpc.aio.insecure_channel(INTERPRETER_ENDPOINT) as ch:
            # gRPC 스텁 생성 (서버의 메서드를 호출할 수 있는 클라이언트 객체)
            stub = voc_pb2_grpc.InterpreterStub(ch)
            # ParseQuestion RPC 호출: 자연어 질의를 파싱하여 구조화된 정보 추출
            res = await stub.ParseQuestion(
                voc_pb2.ParseQuestionReq(
                    question=question,      # 사용자의 자연어 질의
                    default_csv=final_csv   # 기본 CSV 경로 전달 (문자열 "default_csv" 방지)
                ), timeout=timeout  # 타임아웃 설정
            )

        # ============ Intent 딕셔너리 구성 ============
        # Interpreter가 반환한 결과를 딕셔너리 형태로 정리합니다
        intent = {
            "task":      res.task or "both",           # 작업 유형: "summary", "policy", "both"
            "filters":   list(res.filters),            # 필터 키워드 리스트
            "max_items": res.max_items or 30,          # 최대 분석 항목 수 (기본값: 30)
            "csv_path":  res.csv_path or final_csv,  # 최종 CSV 경로 (interpreter가 준 값 우선)
            "needs_clarification": res.needs_clarification,
            "clarifying_question": res.clarifying_question or "",
            "intent_enriched": res.intent_enriched,
            "history_evidence": list(res.history_evidence),
        }
        agent_event("Orchestrator", "intent_received", **intent)

        if intent["needs_clarification"]:
            message = intent["clarifying_question"]
            agent_event(
                "Orchestrator", "pipeline_paused_for_clarification", message=message
            )
            return {
                "ok": False,
                "summary": "",
                "policy": "",
                "intent_json": json.dumps(intent, ensure_ascii=False),
                "eval_json": "{}",
                "summary_critic_json": "{}",
                "intent_guardrail_json": json.dumps(
                    {
                        "passed": False,
                        "status": "clarification_required",
                        "reasons": ["질의가 모호하고 보강할 유사 VOC 이력이 없습니다."],
                    },
                    ensure_ascii=False,
                ),
                "trace": "clarification_required",
                "message": message,
                "clarifying_question": message,
                "error": "clarification_required",
            }

        # ============ Summarizer RunPipeline 직접 호출 ============
        # Interpreter에서 파싱된 intent를 사용하여 Summarizer의 RunPipeline을 호출합니다
        # Summarizer는 내부적으로 Retriever, Evaluator, Critic, Improver를 A2A 방식으로 호출하고
        # 최종 결과(summary, policy)를 반환합니다
        # 이렇게 하면 A2A 방식을 유지하면서도 최종 결과를 받을 수 있습니다
        agent_event("Orchestrator", "call_summarizer")
        async with grpc.aio.insecure_channel(SUMMARIZER_ENDPOINT) as ch:
            stub = voc_pb2_grpc.SummarizerStub(ch)
            sres = await stub.RunPipeline(
                voc_pb2.RunPipelineReq(
                    csv_path=intent["csv_path"],
                    filters=intent["filters"],
                    max_items=intent["max_items"],
                    task=intent["task"],
                ),
                timeout=timeout
            )

        agent_event("Orchestrator", "pipeline_completed", ok=sres.ok,
                    summary_length=len(sres.summary), policy_length=len(sres.policy))
        result = {
            "ok": sres.ok,
            "summary": sres.summary or "",
            "policy": sres.policy or "",
            "intent_json": json.dumps(intent, ensure_ascii=False),
            "eval_json": sres.eval_json or "{}",
            "summary_critic_json": sres.summary_critic_json or "{}",
            "trace": sres.trace or "",
            "message": "Pipeline completed via agent-to-agent calls",
        }
        return self._apply_intent_topic_guardrail(intent, result)

    # ============ 파라미터 기반 실행 메서드 ============
    # 자연어 질의 없이 직접 파라미터를 지정하여 VOC 분석을 수행합니다
    # 이 메서드는 Interpreter를 거치지 않고 바로 파이프라인을 실행합니다
    async def run_with_params(
        self,
        filters: Optional[List[str]],
        task: str,
        max_items: int,
        csv_path: str,
        timeout: float = 180.0
    ) -> Dict[str, Any]:
        """
        직접 파라미터를 지정하여 VOC 분석 파이프라인을 실행합니다.
        
        Args:
            filters: 필터링할 키워드 리스트 (None이면 필터링 없음)
            task: 수행할 작업 ("summary", "policy", "both")
            max_items: 분석할 최대 VOC 개수
            csv_path: VOC 데이터 CSV 파일 경로
            timeout: 각 gRPC 호출의 타임아웃 시간(초)
            
        Returns:
            Dict: 분석 결과 (summary, policy, trace 등 포함)
        """

        # CSV 경로 우선순위 결정
        final_csv = csv_path or DEFAULT_CSV
        agent_event("Orchestrator", "pipeline_started", mode="params",
                    task=task, filters=filters or [], max_items=max_items,
                    csv_path=final_csv)

        retriever_error = await self._check_retriever(timeout)
        if retriever_error:
            return retriever_error

        # ============ Intent 딕셔너리 구성 ============
        # 사용자가 제공한 파라미터를 intent 딕셔너리로 구성합니다
        intent = {
            "task": task or "both",                    # 작업 유형 (기본값: "both")
            "filters": filters or [],                  # 필터 리스트 (None이면 빈 리스트)
            "max_items": max_items or 30,              # 최대 항목 수 (기본값: 30)
            "csv_path": final_csv,                     # CSV 경로
        }

        # ============ Summarizer RunPipeline 직접 호출 ============
        # A2A 방식을 유지하면서 결과를 받기 위해 Summarizer의 RunPipeline을 호출합니다
        # Summarizer는 내부적으로 Retriever, Evaluator, Critic, Improver를 A2A 방식으로 호출하고
        # 최종 결과(summary, policy)를 반환합니다
        agent_event("Orchestrator", "call_summarizer")
        async with grpc.aio.insecure_channel(SUMMARIZER_ENDPOINT) as ch:
            stub = voc_pb2_grpc.SummarizerStub(ch)
            sres = await stub.RunPipeline(
                voc_pb2.RunPipelineReq(
                    csv_path=final_csv,
                    filters=filters or [],
                    max_items=max_items,
                    task=task or "both",
                ),
                timeout=timeout
            )

        agent_event("Orchestrator", "pipeline_completed", ok=sres.ok,
                    summary_length=len(sres.summary), policy_length=len(sres.policy))
        return {
            "ok": sres.ok,
            "summary": sres.summary or "",
            "policy": sres.policy or "",
            "intent_json": json.dumps(intent, ensure_ascii=False),
            "eval_json": sres.eval_json or "{}",
            "summary_critic_json": sres.summary_critic_json or "{}",
            "trace": sres.trace or "",
            "message": "Pipeline completed via agent-to-agent calls",
        }

