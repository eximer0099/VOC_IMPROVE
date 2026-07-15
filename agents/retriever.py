# =============================================================
# File: retriever.py
# Port: 6002
# Role: VOC CSV에서 필터 기반 텍스트 검색 (OR 검색)
# =============================================================

# ============ 표준 라이브러리 및 외부 패키지 임포트 ============
# 비동기 프로그래밍 지원
import asyncio
# 운영체제 관련 기능 (파일 존재 여부 확인 등)
import os
# gRPC 라이브러리 (비동기 서버 통신)
import grpc
# CSV 파일 읽기/쓰기 지원
import csv
import re

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
from utils.text_normalization import build_search_terms, normalize_korean_text


# ============ Retriever Agent 비즈니스 로직 ============
# CSV 파일에서 필터 조건에 맞는 VOC 데이터를 검색하는 에이전트
# OR 검색 방식: 필터 키워드 중 하나라도 포함되면 결과에 포함됩니다
# -------------------------------------------------------------
# Retriever Agent Logic (OR 검색)
# -------------------------------------------------------------
class RetrieverAgent:
    """
    filters 기반으로 VOC CSV에서 텍스트를 추출하는 Agent.
    OR 검색(any): filters 중 하나라도 포함되면 결과로 포함.
    max_items 개수만큼만 반환한다.
    """

    # ============ 초기화 메서드 ============
    def __init__(self):
        """
        RetrieverAgent 인스턴스를 초기화합니다.
        """
        self._csv_write_lock = asyncio.Lock()

    @staticmethod
    def _search_csv(csv_path: str, filters: list[str], max_items: int) -> list[str]:
        """CSV를 매번 새로 읽어 필터와 일치하는 행을 반환합니다."""
        results: list[str] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            for row in csv.reader(fp):
                line = normalize_korean_text(" ".join(row))
                if not filters or any(value in line for value in filters):
                    results.append(" ".join(row))
                    if len(results) >= max_items:
                        break
        return results

    @staticmethod
    def _next_customer_id(csv_path: str) -> str:
        """기존 CUST 번호의 최댓값 다음 고객 ID를 반환합니다."""
        highest = 0
        width = 3
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            for row in csv.reader(fp):
                if not row:
                    continue
                match = re.fullmatch(r"CUST(\d+)", row[0].strip(), re.IGNORECASE)
                if match:
                    highest = max(highest, int(match.group(1)))
                    width = max(width, len(match.group(1)))
        return f"CUST{highest + 1:0{width}d}"

    async def _append_missing_input(self, csv_path: str, content: str) -> str:
        """검색되지 않은 입력을 고객 ID와 함께 CSV에 추가합니다."""
        async with self._csv_write_lock:
            customer_id = self._next_customer_id(csv_path)
            with open(csv_path, "a", encoding="utf-8", newline="") as fp:
                csv.writer(fp).writerow([customer_id, content])
        agent_event(
            "Retriever", "missing_input_added", csv_path=csv_path,
            customer_id=customer_id, content=content,
        )
        return customer_id

    # ============ 검색 실행 메서드 ============
    async def run(self, csv_path: str, filters: list[str], max_items: int) -> list[str]:
        """
        CSV 파일에서 필터 조건에 맞는 VOC 텍스트를 검색합니다.
        
        OR 검색 방식: filters 리스트의 키워드 중 하나라도 포함되면 결과에 포함됩니다.
        검색은 대소문자를 구분하지 않습니다 (소문자로 변환하여 비교).
        
        Args:
            csv_path: 검색할 CSV 파일 경로
            filters: 필터링할 키워드 리스트 (빈 리스트면 필터링 없음)
            max_items: 최대 반환할 항목 수 (1~500 범위로 제한)
            
        Returns:
            list[str]: 검색된 VOC 텍스트 리스트
            
        Raises:
            FileNotFoundError: CSV 파일이 존재하지 않을 때
        """
        # ============ CSV 파일 존재 여부 확인 ============
        # 파일이 없으면 조기 종료하여 불필요한 처리를 방지합니다
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"VOC 데이터 파일 오류: 파일을 찾을 수 없습니다 ({csv_path}). "
                "voc.csv 경로 또는 A2A_VOC_CSV 환경변수를 확인하세요."
            )

        # ============ 필터 전처리 ============
        # 필터 키워드들을 소문자로 변환하고 앞뒤 공백을 제거합니다
        # 빈 문자열은 제외합니다
        input_parts = [str(f).strip() for f in (filters or []) if str(f).strip()]
        primary_filters = [
            normalized
            for value in input_parts
            if (normalized := normalize_korean_text(value))
        ]
        expanded_filters = build_search_terms(input_parts)
        filters = primary_filters

        # ============ max_items 검증 및 제한 ============
        # max_items를 정수로 변환하고 유효한 범위로 제한합니다
        try:
            agent_event("Retriever", "retrieve", csv_path=csv_path,
                        filters=filters, max_items=max_items)
            max_items = int(max_items)
        except Exception:
            # 변환 실패 시 기본값 사용
            max_items = 30

        # ============ 범위 제한 ============
        # 최소값: 30 (0 이하일 때)
        if max_items <= 0:
            max_items = 30
        # 최대값: 500 (너무 많은 항목 반환 방지)
        if max_items > 500:
            max_items = 500

        # 전체 구문 검색을 우선하여 정밀도를 유지하고, 결과가 없을 때만
        # 오탈자/구어체에서 추출한 핵심 토큰으로 재검색합니다.
        results = self._search_csv(csv_path, primary_filters, max_items)
        if not results and expanded_filters != primary_filters:
            filters = expanded_filters
            results = self._search_csv(csv_path, filters, max_items)
            if results:
                agent_event(
                    "Retriever",
                    "normalized_fallback_match",
                    primary_filters=primary_filters,
                    expanded_filters=expanded_filters,
                    retrieved_count=len(results),
                )

        # ============ 결과 반환 ============
        if not results:
            agent_event(
                "Retriever",
                "no_related_data",
                csv_path=csv_path,
                filters=filters,
                retrieved_count=0,
                message="관련 데이터 없음: 입력 질문과 일치하는 VOC를 찾지 못했습니다.",
            )

            # RetrieveReq에는 원문 질문 필드가 없으므로 필터 원문을 입력
            # 내용으로 간주합니다. 추가한 뒤 CSV를 다시 읽어 새 행을
            # 데이터 원본에서 참조하도록 합니다.
            input_content = " ".join(input_parts).strip()
            if input_content:
                await self._append_missing_input(csv_path, input_content)
                results = self._search_csv(csv_path, filters, max_items)
                agent_event(
                    "Retriever",
                    "retrieve_after_append",
                    csv_path=csv_path,
                    retrieved_count=len(results),
                )

        agent_file_event(
            "Retriever",
            "output",
            operation="retrieve",
            texts=results,
            retrieved_count=len(results),
        )
        return results


# ============ gRPC 서비스 구현 ============
# Protocol Buffers로 정의된 서비스를 구현하는 클래스
# 클라이언트의 RPC 요청을 받아 RetrieverAgent의 비즈니스 로직을 실행합니다
# -------------------------------------------------------------
# gRPC Servicer
# -------------------------------------------------------------
class RetrieverServicer(voc_pb2_grpc.RetrieverServicer):
    """
    Retriever gRPC 서비스를 구현하는 클래스입니다.
    
    voc_pb2_grpc.RetrieverServicer를 상속받아
    Protocol Buffers로 정의된 RPC 메서드들을 구현합니다.
    """

    # ============ 초기화 메서드 ============
    def __init__(self):
        """
        RetrieverServicer 인스턴스를 초기화합니다.
        비즈니스 로직을 담당하는 RetrieverAgent를 생성합니다.
        """
        self.agent = RetrieverAgent()

    # ============ Retrieve RPC 구현 ============
    @log_response_time("Retriever")
    async def Retrieve(self, request, context):
        """
        Retrieve RPC를 구현합니다.
        
        클라이언트로부터 CSV 경로, 필터, 최대 항목 수를 받아
        필터 조건에 맞는 VOC 텍스트만 반환합니다.
        
        Args:
            request: RetrieveReq 메시지 (csv_path, filters, max_items, task 포함)
            context: gRPC 서비스 컨텍스트 (에러 처리 등에 사용)
            
        Returns:
            RetrieveRes: 검색된 텍스트 리스트를 포함한 응답 메시지
        """
        try:
            # ============ 요청 파라미터 추출 ============
            csv_path = request.csv_path        # CSV 파일 경로
            filters = list(request.filters)    # 필터 키워드 리스트 (gRPC repeated 필드를 리스트로 변환)
            max_items = request.max_items       # 최대 검색 항목 수
            # ============ 검색 실행 ============
            # 에이전트의 run 메서드를 호출하여 VOC 데이터를 검색합니다
            texts = await self.agent.run(csv_path, filters, max_items)
            texts = [str(t).strip() for t in texts if str(t).strip()]
            agent_event("Retriever", "completed", retrieved_count=len(texts))

            # ============ 응답 메시지 생성 및 반환 ============
            # 검색된 텍스트를 gRPC 응답 메시지로 감싸서 반환합니다
            return voc_pb2.RetrieveRes(texts=texts)

        except FileNotFoundError as e:
            agent_event(
                "Retriever", "data_file_error", csv_path=request.csv_path, error=str(e)
            )
            await context.abort(grpc.StatusCode.NOT_FOUND, str(e))

        except Exception as e:
            log_authentication_error("Retriever", e)
            agent_event("Retriever", "error", error=str(e))
            # ============ 에러 처리 ============
            # 예외 발생 시 gRPC 에러로 변환하여 클라이언트에 전달합니다
            await context.abort(
                grpc.StatusCode.INTERNAL,  # 내부 서버 오류 상태 코드
                f"Retriever error: {e}"   # 에러 메시지
            )


# ============ gRPC 서버 실행 함수 ============
# 이 모듈을 직접 실행할 때 gRPC 서버를 시작하는 함수
# -------------------------------------------------------------
# gRPC Server
# -------------------------------------------------------------
async def serve():
    """
    Retriever gRPC 서버를 시작합니다.
    
    환경변수 RETRIEVER_ENDPOINT에서 엔드포인트를 읽어옵니다.
    기본값은 "0.0.0.0:6002"입니다 (모든 네트워크 인터페이스의 6002 포트).
    """
    # ============ 엔드포인트 설정 ============
    # 환경변수에서 엔드포인트를 읽어오고, 없으면 기본값을 사용합니다
    endpoint = os.environ.get("RETRIEVER_ENDPOINT", "0.0.0.0:6002")

    # ============ gRPC 서버 생성 ============
    # 비동기 gRPC 서버 인스턴스를 생성합니다
    server = grpc.aio.server()
    # ============ 서비스 등록 ============
    # RetrieverServicer를 서버에 등록하여 RPC 요청을 처리할 수 있도록 합니다
    voc_pb2_grpc.add_RetrieverServicer_to_server(RetrieverServicer(), server)
    # ============ 포트 바인딩 ============
    # 서버를 지정된 엔드포인트에 바인딩합니다 (TLS 없이)
    bind_agent_port(server, endpoint, "Retriever")

    # ============ 서버 시작 로그 ============
    # 서버가 시작되었음을 콘솔에 출력합니다
    agent_event("Retriever", "server_started", bind=endpoint)

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
