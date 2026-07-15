# ================================================================
# File: summarizer.py
# Port: 6003
# Role: 요약 후보 생성 + refine 수행
# ================================================================

# ============ 표준 라이브러리 및 외부 패키지 임포트 ============
# 비동기 프로그래밍 지원
import asyncio
# 운영체제 관련 기능 (환경변수 읽기 등)
import os
# JSON 데이터 처리
import json
# 근거 인용 형식 및 공백 정규화
import re
# gRPC 라이브러리 (비동기 서버 통신)
import grpc

# ============ Protocol Buffers 생성 파일 임포트 ============
# voc.proto 파일로부터 생성된 메시지 및 서비스 정의
import voc_pb2
import voc_pb2_grpc
from utils.agent_log import (
    agent_event,
    agent_file_event,
    log_authentication_error,
    log_response_time,
)
from grpc_server import bind_agent_port

# ============ 프로젝트 내부 모듈 임포트 ============
# OpenAI Chat API를 사용하기 위한 래퍼 클래스
from llm_wrappers.openai_chat import OpenAIChat


# ============ Summarizer Agent 비즈니스 로직 ============
# VOC 텍스트를 요약하고 여러 후보를 생성하며, 필요시 개선하는 에이전트
# ---------------------------------------------------------------
# Summarizer Agent Logic
# ---------------------------------------------------------------
class SummarizerAgent:
    """
    VOC 텍스트를 요약하고 후보(S0,S1,S2...)를 생성하며,
    필요한 경우 refine도 처리하는 agent.
    """

    _EVIDENCE_PATTERN = re.compile(
        r"^(?P<summary>.+?)\s*\|\s*근거:\s*\[VOC(?P<index>\d+)\]\s*(?P<quote>.+)$",
        re.DOTALL,
    )

    @staticmethod
    def _normalize_grounding_text(value: str) -> str:
        return " ".join(str(value).split()).strip()

    @classmethod
    def _is_grounded_candidate(cls, candidate: str, texts: list[str]) -> bool:
        """후보 요약과 인용문이 지정된 VOC 원문에 직접 포함되는지 확인한다."""
        match = cls._EVIDENCE_PATTERN.fullmatch(str(candidate).strip())
        if not match:
            return False

        index = int(match.group("index")) - 1
        if index < 0 or index >= len(texts):
            return False

        source = cls._normalize_grounding_text(texts[index])
        quote = cls._normalize_grounding_text(match.group("quote"))
        summary = cls._normalize_grounding_text(match.group("summary"))
        return bool(summary and quote and quote == source and summary in source)

    @classmethod
    def _fallback_grounded_candidate(cls, source: str, index: int) -> str:
        normalized = cls._normalize_grounding_text(source)
        return f"{normalized} | 근거: [VOC{index + 1}] {normalized}"

    @classmethod
    def grounding_check(
        cls,
        candidates: dict[str, str],
        texts: list[str],
        expected_count: int,
    ) -> dict[str, str]:
        """모든 후보를 원문과 대조하고 실패한 후보를 원문 발췌로 교체한다."""
        if not texts:
            return candidates

        grounded: dict[str, str] = {}
        replaced: list[str] = []
        for position in range(max(1, expected_count)):
            key = f"S{position}"
            candidate = str(candidates.get(key, "")).strip()
            if candidate and cls._is_grounded_candidate(candidate, texts):
                grounded[key] = candidate
                continue

            source_index = position % len(texts)
            grounded[key] = cls._fallback_grounded_candidate(
                texts[source_index], source_index
            )
            replaced.append(key)

        agent_event(
            "Summarizer",
            "grounding_check",
            checked_count=len(grounded),
            replaced_keys=replaced,
            grounded=not replaced,
        )
        return grounded

    # ============ 초기화 메서드 ============
    def __init__(self):
        """
        SummarizerAgent 인스턴스를 초기화합니다.
        LLM 래퍼를 인스턴스 변수로 저장하여 재사용합니다.
        """
        # 클래스 변수가 아닌 인스턴스 변수로 생성해야 각 요청마다 독립적인 상태를 유지할 수 있습니다
        self.llm = OpenAIChat()
        # 다음 에이전트 엔드포인트 설정
        self.evaluator_endpoint = os.environ.get("EVALUATOR_ENDPOINT", "localhost:6004")
        # run_pipeline에서 사용하는 에이전트 엔드포인트 설정
        self.retriever_endpoint = os.environ.get("RETRIEVER_ENDPOINT", "localhost:6002")
        self.critic_endpoint = os.environ.get("CRITIC_ENDPOINT", "localhost:6005")
        self.improver_endpoint = os.environ.get("IMPROVER_ENDPOINT", "localhost:6006")

    # ============ 요약 후보 생성 메서드 ============
    async def make_candidates(self, texts: list[str], max_items: int, n: int):
        """
        여러 개의 요약 후보를 생성합니다.
        
        LLM을 사용하여 동일한 VOC 데이터로부터 다양한 관점의 요약을 생성합니다.
        여러 후보를 생성하는 이유: Evaluator가 비교 평가하여 최적의 요약을 선택하기 위함입니다.
        
        Args:
            texts: 요약할 VOC 텍스트 리스트
            max_items: 최대 사용할 텍스트 개수 (메모리 및 토큰 제한 고려)
            n: 생성할 후보 개수 (일반적으로 3개)
            
        Returns:
            dict: 후보 키(S0, S1, S2 등)와 요약 텍스트의 딕셔너리
        """
        # ============ 입력 텍스트 정규화 ============
        # 빈 문자열과 None을 제거하고, 공백만 있는 항목은 제외합니다
        texts = [str(t).strip() for t in (texts or []) if str(t).strip()]
        if not texts:
            candidates = {"S0": "요약할 수 있는 VOC 본문이 비어 있습니다."}
            agent_file_event(
                "Summarizer", "output", operation="make_candidates",
                candidates=candidates,
            )
            return candidates

        # ============ 최대 개수 계산 ============
        # max_items가 0 또는 비정상 값이면 전체 텍스트 수를 사용합니다
        # 이렇게 해야 Retriever가 1건만 찾았는데도 요약 프롬프트가 빈 문자열로 시작하지 않습니다
        try:
            effective_max_items = int(max_items or len(texts))
        except Exception:
            effective_max_items = len(texts)

        effective_max_items = max(1, min(effective_max_items, len(texts)))

        # ============ 텍스트 결합 ============
        # 여러 VOC 텍스트를 줄바꿈으로 구분하여 하나의 문자열로 결합합니다
        # max_items 개수만큼만 사용하여 토큰 제한을 준수합니다
        source_texts = texts[:effective_max_items]
        joined = "\n".join(
            f"[VOC{index}] {text}"
            for index, text in enumerate(source_texts, start=1)
        )

        # ============ 프롬프트 구성 ============
        # LLM에게 요약 후보를 생성하도록 지시하는 프롬프트를 작성합니다
        # 형식: S0, S1, S2 등의 키와 함께 요약을 출력하도록 명시합니다
        prompt = f"""
다음 VOC 원문만 사용하여 발췌형 요약 후보를 {n}개 생성해라.
원문에 없는 사실, 원인, 수치, 날짜, 조직, 해결책은 절대 추가하지 마라.
요약 본문은 선택한 VOC 원문의 연속된 일부 문구를 표현 변경 없이 그대로 사용해라.
각 후보에는 반드시 실제 원문 한 건을 전체 그대로 근거로 인용해라.
형식:
S0: <원문에서 그대로 발췌한 요약> | 근거: [VOC번호] <해당 VOC 원문 전체>
S1: <원문에서 그대로 발췌한 요약> | 근거: [VOC번호] <해당 VOC 원문 전체>
S2: <원문에서 그대로 발췌한 요약> | 근거: [VOC번호] <해당 VOC 원문 전체>

근거 인용은 입력의 VOC 번호와 원문을 한 글자도 바꾸지 말고 복사해라.
설명, 머리말, 코드 블록은 출력하지 마라.

데이터:
{joined}
"""

        # ============ LLM 호출 ============
        # 비동기로 LLM을 호출하여 요약 후보를 생성합니다
        result = await self.llm(prompt)   
        # ============ 후보 파싱 및 반환 ============
        # LLM 응답에서 후보들을 파싱하여 딕셔너리 형태로 반환합니다
        candidates = self._parse_candidates(result)
        # 후보의 요약과 근거 인용을 실제 Retriever 원문과 대조합니다.
        # 검증 실패 후보는 원문 발췌형 후보로 교체하여 환각 전파를 차단합니다.
        candidates = self.grounding_check(candidates, source_texts, n)
        agent_file_event(
            "Summarizer", "output", operation="make_candidates",
            candidates=candidates,
        )
        return candidates

    # ============ 요약 개선 메서드 ============
    async def refine(
        self,
        draft: str,
        edits_json: str,
        source_texts: list[str] | None = None,
    ):
        """
        Critic이 제안한 edits 기반으로 요약문을 개선(refine)합니다.
        
        Critic이 요약의 품질을 검토하고 수정 지침을 제공하면,
        이 메서드를 사용하여 원본 요약을 개선합니다.
        
        Args:
            draft: 개선할 원본 요약 텍스트
            edits_json: Critic이 제공한 수정 지침 (JSON 문자열)
            
        Returns:
            str: 개선된 요약 텍스트 (앞뒤 공백 제거)
        """
        if not source_texts:
            # 원문이 없으면 생성 결과를 대조할 수 없으므로 새 내용을 만들지 않습니다.
            agent_event("Summarizer", "refine_skipped_without_sources")
            return draft.strip()

        # ============ 프롬프트 구성 ============
        # 원본 요약과 수정 지침을 포함한 프롬프트를 작성합니다
        # LLM에게 수정 지침에 따라 요약을 개선하도록 지시합니다
        prompt = f"""
아래 draft 요약문을 edits 지시에 따라 개선해라.
VOC 원문에 없는 사실, 원인, 수치, 날짜, 조직, 해결책은 추가하지 마라.
출력은 반드시 '<원문 발췌> | 근거: [VOC번호] <해당 VOC 원문 전체>' 형식으로 작성해라.

draft:
{draft}

edits:
{edits_json}

VOC 원문:
{chr(10).join(f'[VOC{i}] {text}' for i, text in enumerate(source_texts or [], 1))}

출력: 개선된 요약문만 제공
"""

        # ============ LLM 호출 및 결과 반환 ============
        # 비동기로 LLM을 호출하여 개선된 요약을 생성합니다
        result = await self.llm(prompt)
        # 앞뒤 공백을 제거하여 깔끔한 텍스트를 반환합니다
        refined = result.strip()
        checked = self.grounding_check({"S0": refined}, source_texts, 1)["S0"]
        if checked != refined:
            agent_event("Summarizer", "refine_grounding_rejected")
            refined = draft
        agent_file_event(
            "Summarizer", "output", operation="refine", text=refined
        )
        return refined

    # ============ 헬퍼 메서드 ============
    # 내부적으로 사용하는 유틸리티 함수들
    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _parse_candidates(self, text: str) -> dict:
        """
        LLM 응답 텍스트에서 요약 후보를 파싱합니다.
        
        LLM이 "S0: ...", "S1: ..." 형식으로 출력한 텍스트에서
        후보 키와 요약 텍스트를 추출하여 딕셔너리로 변환합니다.
        
        Args:
            text: LLM 응답 텍스트
            
        Returns:
            dict: 후보 키(S0, S1 등)와 요약 텍스트의 딕셔너리
                 파싱 실패 시 전체 텍스트를 S0로 반환
        """
        # ============ 줄 단위 분리 ============
        # 텍스트를 줄 단위로 분리하여 각 줄을 처리합니다
        lines = text.splitlines()
        # ============ 후보 딕셔너리 초기화 ============
        out = {}
        # ============ 각 줄 파싱 ============
        # 각 줄에서 "키: 값" 형식을 찾아 후보를 추출합니다
        current_key = None
        for line in lines:
            if ":" in line:
                # 콜론을 기준으로 키와 값을 분리합니다
                k, v = line.split(":", 1)  # maxsplit=1로 첫 번째 콜론만 분리
                k = k.strip()  # 키의 앞뒤 공백 제거
                v = v.strip()  # 값의 앞뒤 공백 제거
                # S로 시작하는 키만 후보로 인정합니다 (S0, S1, S2 등)
                if k.startswith("S") and k[1:].isdigit():
                    out[k] = v
                    current_key = k
                    continue
            # Preserve multiline candidate content instead of silently dropping it.
            if current_key and line.strip():
                out[current_key] = f"{out[current_key]}\n{line.strip()}".strip()
        # ============ 폴백 처리 ============
        # 파싱된 후보가 없으면 전체 텍스트를 S0로 사용합니다
        if not out:
            out = {"S0": text.strip()}
        # ============ 결과 반환 ============
        return out

    # ============ 전체 요약 파이프라인 실행 메서드 ============
    async def run_pipeline(
        self,
        csv_path: str,
        filters: list[str],
        max_items: int,
        task: str,
        timeout: float = 180.0,
    ) -> dict:
        """
        요약 생성 전체 파이프라인을 실행합니다.
        Retriever, Evaluator, Critic, Improver를 정해진 순서로 한 번씩 호출합니다.
        
        Args:
            csv_path: CSV 파일 경로
            filters: 필터 키워드 리스트
            max_items: 최대 항목 수
            task: 작업 유형 ("summary", "policy", "both")
            timeout: gRPC 호출 타임아웃
            
        Returns:
            dict: 요약 결과 및 추적 정보
        """
        trace = []
        agent_event("Summarizer", "pipeline_started", task=task,
                    filters=filters, max_items=max_items)
        
        # ============ 1단계: Retriever 호출 ============
        agent_event("Summarizer", "call_retriever")
        async with grpc.aio.insecure_channel(self.retriever_endpoint) as ch:
            stub = voc_pb2_grpc.RetrieverStub(ch)
            rres = await stub.Retrieve(
                voc_pb2.RetrieveReq(
                    csv_path=csv_path,
                    filters=filters,
                    max_items=max_items,
                ),
                timeout=timeout
            )
        texts = list(rres.texts)
        trace.append(f"retrieved={len(texts)}")
        agent_event("Summarizer", "retrieval_received", count=len(texts))
        
        if not texts:
            return {
                "summary": "",
                "trace": "; ".join(trace),
                "ok": False,
            }
        
        # ============ 2단계: 요약 후보 생성 ============
        candidates = await self.make_candidates(texts, max_items, n=3)
        trace.append(f"candidates={list(candidates.keys())}")
        agent_event("Summarizer", "candidates_created", keys=list(candidates.keys()))
        
        # ============ 3단계: Evaluator 호출 ============
        agent_event("Summarizer", "call_evaluator")
        async with grpc.aio.insecure_channel(self.evaluator_endpoint) as ch:
            stub = voc_pb2_grpc.EvaluatorStub(ch)
            eres = await stub.Evaluate(
                voc_pb2.EvaluateReq(
                    task=task,
                    candidates=candidates
                ),
                timeout=timeout
            )
        
        winner_key = eres.winner or sorted(candidates.keys())[0]
        summary = candidates.get(winner_key) or next(
            (text for text in candidates.values() if text.strip()), ""
        )
        if not summary:
            return {
                "summary": "",
                "trace": "; ".join(trace + [f"winner={winner_key}", "empty_candidates"]),
                "ok": False,
            }
        eval_json = eres.scores_json or "{}"
        trace.append(f"winner={winner_key}")
        agent_event("Summarizer", "winner_selected", winner=winner_key)
        
        # ============ 4단계: Critic 호출 ============
        agent_event("Summarizer", "call_critic", role="summary")
        async with grpc.aio.insecure_channel(self.critic_endpoint) as ch:
            stub = voc_pb2_grpc.CriticStub(ch)
            cres = await stub.Review(
                voc_pb2.ReviewReq(
                    doc=summary,
                    role="summary"
                ),
                timeout=timeout
            )
        
        summary_critic_info = {
            "need_refine": cres.need_refine,
            "edits": list(cres.edits),
            "ask_more_samples": cres.ask_more_samples,
            "revalidated": False,
            "feedback_applied": None,
            "remaining_edits": [],
        }
        
        # ============ 5단계: 필요시 요약 개선 ============
        if cres.need_refine and cres.edits:
            agent_event("Summarizer", "refine_summary", edits=list(cres.edits))
            summary = await self.refine(
                summary,
                json.dumps({"edits": list(cres.edits)}, ensure_ascii=False),
                source_texts=texts,
            )
            trace.append("summary_refined")

            # ============ Critic 피드백 반영 재검증 ============
            # 이전 edits와 수정 결과를 함께 전달하여 실제 반영 여부를 한 번 확인합니다.
            revalidation_input = json.dumps(
                {
                    "previous_edits": list(cres.edits),
                    "revised_summary": summary,
                },
                ensure_ascii=False,
            )
            agent_event(
                "Summarizer",
                "call_critic_revalidation",
                previous_edit_count=len(cres.edits),
            )
            async with grpc.aio.insecure_channel(self.critic_endpoint) as ch:
                stub = voc_pb2_grpc.CriticStub(ch)
                recheck = await stub.Review(
                    voc_pb2.ReviewReq(
                        doc=revalidation_input,
                        role="summary_revalidation",
                    ),
                    timeout=timeout,
                )

            feedback_applied = not recheck.need_refine
            summary_critic_info.update(
                {
                    "revalidated": True,
                    "feedback_applied": feedback_applied,
                    "remaining_edits": list(recheck.edits),
                    "revalidation_ask_more_samples": recheck.ask_more_samples,
                }
            )
            trace.append(
                "critic_feedback_applied"
                if feedback_applied
                else "critic_feedback_not_applied"
            )
            agent_event(
                "Summarizer",
                "critic_revalidation_completed",
                feedback_applied=feedback_applied,
                remaining_edits=list(recheck.edits),
            )

            if not feedback_applied:
                # 원문 grounding을 통과하지 못하는 Critic 요구 때문에 정책 생성까지
                # 중단하지 않습니다. 미반영 지침은 감사 정보에 남기고, 마지막으로
                # 검증된 요약을 Improver에 전달합니다.
                summary_critic_info["continued_with_grounded_summary"] = True
                trace.append("continued_with_grounded_summary")
                agent_event(
                    "Summarizer",
                    "critic_feedback_unresolved_continuing",
                    reason="use_last_grounded_summary",
                )
        
        # ============ 6단계: 최종 요약으로 Improver 호출 ============
        # 정책이 필요한 경우에만 모든 검토와 수정을 마친 요약으로 한 번 호출합니다.
        policy = ""
        if task in ("policy", "both"):
            agent_event("Summarizer", "call_improver")
            async with grpc.aio.insecure_channel(self.improver_endpoint) as ch:
                stub = voc_pb2_grpc.ImproverStub(ch)
                pres = await stub.Improve(
                    voc_pb2.PolicyReq(summary=summary),
                    timeout=timeout,
                )
            policy = pres.policy or ""
            trace.append("policy_created")
            agent_event("Summarizer", "policy_received", length=len(policy))
        
        agent_event("Summarizer", "pipeline_completed", ok=True)
        output = {
            "summary": summary,
            "policy": policy,
            "eval_json": eval_json,
            "summary_critic_json": json.dumps(summary_critic_info, ensure_ascii=False),
            "trace": "; ".join(trace),
            "ok": True,
        }
        agent_file_event(
            "Summarizer", "output", operation="run_pipeline", **output
        )
        return output


# ============ gRPC 서비스 구현 ============
# Protocol Buffers로 정의된 서비스를 구현하는 클래스
# 각 RPC 메서드는 클라이언트의 요청을 받아 비즈니스 로직을 실행합니다
# ---------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------
class SummarizerServicer(voc_pb2_grpc.SummarizerServicer):
    """
    Summarizer gRPC 서비스를 구현하는 클래스입니다.
    
    voc_pb2_grpc.SummarizerServicer를 상속받아
    Protocol Buffers로 정의된 RPC 메서드들을 구현합니다.
    """

    # ============ 초기화 메서드 ============
    def __init__(self):
        """
        SummarizerServicer 인스턴스를 초기화합니다.
        비즈니스 로직을 담당하는 SummarizerAgent를 생성합니다.
        """
        self.agent = SummarizerAgent()

    # ============ MakeCandidates RPC 구현 ============
    @log_response_time("Summarizer")
    async def MakeCandidates(self, request, context):
        """
        MakeCandidates RPC를 구현합니다.
        
        클라이언트로부터 VOC 텍스트 리스트를 받아
        여러 개의 요약 후보만 생성합니다.
        
        Args:
            request: SummarizeReq 메시지 (texts, max_items, n, task 포함)
            context: gRPC 서비스 컨텍스트 (에러 처리 등에 사용)
            
        Returns:
            SummarizeRes: 생성된 후보 딕셔너리를 포함한 응답 메시지
        """
        try:
            # ============ 요약 후보 생성 ============
            # 에이전트의 make_candidates 메서드를 호출하여 후보를 생성합니다
            candidates = await self.agent.make_candidates(
                texts=list(request.texts),      # gRPC repeated 필드를 리스트로 변환
                max_items=request.max_items,    # 최대 항목 수
                n=request.n,                    # 생성할 후보 개수
            )
            
            # ============ 응답 메시지 생성 및 반환 ============
            # 생성된 후보를 gRPC 응답 메시지로 감싸서 반환합니다
            return voc_pb2.SummarizeRes(candidates=candidates)

        except Exception as e:
            log_authentication_error("Summarizer", e)
            # ============ 에러 처리 ============
            # 예외 발생 시 gRPC 에러로 변환하여 클라이언트에 전달합니다
            await context.abort(
                grpc.StatusCode.INTERNAL,  # 내부 서버 오류 상태 코드
                f"Summarizer.MakeCandidates error: {e}"  # 에러 메시지
            )

    # ============ Refine RPC 구현 ============
    @log_response_time("Summarizer")
    async def Refine(self, request, context):
        """
        Refine RPC를 구현합니다.
        
        클라이언트로부터 원본 요약과 수정 지침을 받아
        개선된 요약을 생성하여 반환합니다.
        
        Args:
            request: RefineReq 메시지 (draft, edits_json 포함)
            context: gRPC 서비스 컨텍스트 (에러 처리 등에 사용)
            
        Returns:
            RefineRes: 개선된 요약 텍스트를 포함한 응답 메시지
        """
        try:
            # ============ 요약 개선 ============
            # 에이전트의 refine 메서드를 호출하여 요약을 개선합니다
            out = await self.agent.refine(
                draft=request.draft,            # 개선할 원본 요약 텍스트
                edits_json=request.edits_json, # 수정 지침 (JSON 문자열)
            )
            # ============ 응답 메시지 생성 및 반환 ============
            # 개선된 요약을 gRPC 응답 메시지로 감싸서 반환합니다
            return voc_pb2.RefineRes(text=out)

        except Exception as e:
            log_authentication_error("Summarizer", e)
            # ============ 에러 처리 ============
            # 예외 발생 시 gRPC 에러로 변환하여 클라이언트에 전달합니다
            await context.abort(
                grpc.StatusCode.INTERNAL,  # 내부 서버 오류 상태 코드
                f"Summarizer.Refine error: {e}"  # 에러 메시지
            )

    # ============ RunPipeline RPC 구현 ============
    @log_response_time("Summarizer")
    async def RunPipeline(self, request, context):
        """
        RunPipeline RPC를 구현합니다.
        
        요약 생성 전체 파이프라인을 실행합니다.
        Retriever, Evaluator, Critic, Improver를 정해진 순서로 호출합니다.
        
        Args:
            request: RunPipelineReq 메시지 (csv_path, filters, max_items, task 포함)
            context: gRPC 서비스 컨텍스트 (에러 처리 등에 사용)
            
        Returns:
            RunPipelineRes: 요약 결과 및 추적 정보를 포함한 응답 메시지
        """
        try:
            # ============ 파이프라인 실행 ============
            # 에이전트의 run_pipeline 메서드를 호출하여 전체 파이프라인을 실행합니다
            result = await self.agent.run_pipeline(
                csv_path=request.csv_path,
                filters=list(request.filters),
                max_items=request.max_items,
                task=request.task or "both",
                timeout=180.0,
            )
            # ============ 응답 메시지 생성 및 반환 ============
            # 파이프라인 실행 결과를 gRPC 응답 메시지로 감싸서 반환합니다
            return voc_pb2.RunPipelineRes(
                ok=result.get("ok", False),
                summary=result.get("summary", ""),
                policy=result.get("policy", ""),
                eval_json=result.get("eval_json", "{}"),
                summary_critic_json=result.get("summary_critic_json", "{}"),
                trace=result.get("trace", ""),
            )

        except Exception as e:
            log_authentication_error("Summarizer", e)
            # ============ 에러 처리 ============
            # 예외 발생 시 gRPC 에러로 변환하여 클라이언트에 전달합니다
            await context.abort(
                grpc.StatusCode.INTERNAL,  # 내부 서버 오류 상태 코드
                f"Summarizer.RunPipeline error: {e}"  # 에러 메시지
            )


# ============ gRPC 서버 실행 함수 ============
# 이 모듈을 직접 실행할 때 gRPC 서버를 시작하는 함수
# ---------------------------------------------------------------
# gRPC Server
# ---------------------------------------------------------------
async def serve():
    """
    Summarizer gRPC 서버를 시작합니다.
    
    환경변수 SUMMARIZER_ENDPOINT에서 엔드포인트를 읽어옵니다.
    기본값은 "0.0.0.0:6003"입니다 (모든 네트워크 인터페이스의 6003 포트).
    """
    # ============ 엔드포인트 설정 ============
    # 환경변수에서 엔드포인트를 읽어오고, 없으면 기본값을 사용합니다
    endpoint = os.environ.get("SUMMARIZER_ENDPOINT", "0.0.0.0:6003")

    # ============ gRPC 서버 생성 ============
    # 비동기 gRPC 서버 인스턴스를 생성합니다
    server = grpc.aio.server()
    # ============ 서비스 등록 ============
    # SummarizerServicer를 서버에 등록하여 RPC 요청을 처리할 수 있도록 합니다
    voc_pb2_grpc.add_SummarizerServicer_to_server(SummarizerServicer(), server)
    # ============ 포트 바인딩 ============
    # 서버를 지정된 엔드포인트에 바인딩합니다 (TLS 없이)
    bind_agent_port(server, endpoint, "Summarizer")

    # ============ 서버 시작 로그 ============
    # 서버가 시작되었음을 콘솔에 출력합니다
    agent_event("Summarizer", "server_started", bind=endpoint)

    # ============ 서버 시작 및 대기 ============
    # 서버를 시작하고 종료 신호를 받을 때까지 대기합니다
    await server.start()
    # 서버가 종료될 때까지 무한 대기합니다 (Ctrl+C로 종료 가능)
    await server.wait_for_termination()


# ============ 메인 실행 블록 ============
# 스크립트가 직접 실행될 때만 서버를 시작합니다
if __name__ == "__main__":
    # asyncio.run()을 사용하여 비동기 서버를 실행합니다
    asyncio.run(serve())
