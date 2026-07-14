# ================================================================
# File: anthropic_chat.py
# Role: Anthropic Claude LLM Wrapper (async)
# ================================================================

# ============ 표준 라이브러리 및 외부 패키지 임포트 ============
# 운영체제 관련 기능 (환경변수 읽기)
import os
# Anthropic 비동기 클라이언트 (API 호출용)
from anthropic import AsyncAnthropic

from utils.settings import ANTHROPIC_API_KEY, MODEL_POLICY


# ============ Anthropic Claude LLM 래퍼 클래스 ============
class AnthropicChat:
    """
    Anthropic Claude Messages API를 사용하기 위한 비동기 래퍼 클래스입니다.
    
    Summarizer / Evaluator / Critic / Improver 등
    모든 Agent가 await self.llm(prompt) 형태로 호출함.
    따라서 __call__(str) -> str 형태를 반드시 구현해야 함.
    
    이 클래스는 함수처럼 호출 가능한 객체(callable)로 동작합니다.
    """

    # ============ 초기화 메서드 ============
    def __init__(self, model: str = None):
        """
        AnthropicChat 인스턴스를 초기화합니다.
        
        Args:
            model: 사용할 Anthropic 모델명 (None이면 환경변수 또는 기본값 사용)
        """
        # ============ 모델명 설정 ============
        # 사용자가 지정한 모델명이 있으면 사용하고,
        # 없으면 환경변수 A2A_MODEL_POLICY을 확인하고,
        # 그것도 없으면 settings.py의 공통 기본값을 사용합니다
        self.model = model or os.environ.get("A2A_MODEL_POLICY", MODEL_POLICY)
        
        # ============ Anthropic 클라이언트 생성 ============
        # 환경변수 ANTHROPIC_API_KEY에서 API 키를 읽어와 클라이언트를 생성합니다
        # API 키가 없으면 클라이언트 생성은 되지만 실제 호출 시 에러가 발생합니다
        self.client = AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY
        )

    # ============ 호출 가능한 객체 구현 ============
    async def __call__(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        클래스를 함수처럼 호출할 수 있도록 하는 메서드입니다.
        
        이 메서드를 통해 Anthropic Messages API를 호출하여
        프롬프트에 대한 응답을 받아옵니다.
        
        Args:
            prompt: LLM에게 전달할 프롬프트 텍스트
            max_tokens: 최대 생성 토큰 수 (기본값: 1024)
            
        Returns:
            str: LLM이 생성한 응답 텍스트
            
        호출 예시:
            llm = AnthropicChat()
            result = await llm("정책 개선안을 제안해줘")
        """
        # ============ API 호출 ============
        # 비동기로 Anthropic Messages API를 호출합니다
        last_response = None
        for _attempt in range(2):
            last_response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = self._extract_text(last_response.content)
            if text:
                return text

        block_types = [
            getattr(block, "type", type(block).__name__)
            for block in (getattr(last_response, "content", None) or [])
        ]
        raise RuntimeError(
            "Anthropic returned no text after retry "
            f"(model={self.model}, stop_reason={getattr(last_response, 'stop_reason', None)}, "
            f"blocks={block_types})"
        )

    @staticmethod
    def _extract_text(content) -> str:
        """Join every text block in a Messages API response."""
        parts = []
        for block in content or []:
            value = getattr(block, "text", None)
            if value is None and isinstance(block, dict):
                value = block.get("text")
            if value and str(value).strip():
                parts.append(str(value).strip())
        return "\n".join(parts).strip()

