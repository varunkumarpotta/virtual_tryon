"""
LLM Service — Optional integration with OmniRoute endpoint.
Used for garment classification, error explanation, etc.
Never for pixel-level try-on generation.
"""
import logging
import httpx
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Optional LLM integration via OmniRoute-compatible endpoint."""

    def __init__(self):
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        if settings.anthropic_base_url and settings.anthropic_auth_token:
            self._available = True
            logger.info(f"LLM service configured: {settings.anthropic_base_url}")
        else:
            logger.info("LLM service not configured (optional)")

    @property
    def available(self) -> bool:
        return self._available

    async def classify_garment(self, description: str) -> Optional[str]:
        """Use LLM to classify garment type from description."""
        if not self._available:
            return None

        try:
            prompt = (
                f"Classify this clothing item into exactly one category: "
                f"shirt, t-shirt, jacket, pants, jeans, dress, skirt, saree, other.\n"
                f"Item: {description}\n"
                f"Respond with only the category name, nothing else."
            )
            response = await self._chat(prompt)
            if response:
                category = response.strip().lower()
                valid = {"shirt", "t-shirt", "jacket", "pants", "jeans", "dress", "skirt", "saree", "other"}
                return category if category in valid else None
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
        return None

    async def explain_error(self, error: str, context: str = "") -> Optional[str]:
        """Use LLM to generate user-friendly error explanation."""
        if not self._available:
            return None

        try:
            prompt = (
                f"You are helping a user with a virtual try-on application. "
                f"Explain this error in simple, friendly terms and suggest a fix:\n"
                f"Error: {error}\n"
                f"Context: {context}\n"
                f"Keep response under 2 sentences."
            )
            return await self._chat(prompt)
        except Exception as e:
            logger.warning(f"LLM error explanation failed: {e}")
        return None

    async def _chat(self, prompt: str) -> Optional[str]:
        """Send a chat request to the OmniRoute endpoint."""
        if not self._available:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.anthropic_base_url}/v1/messages",
                    headers={
                        "x-api-key": settings.anthropic_auth_token,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": settings.anthropic_model or "claude-sonnet-4-20250514",
                        "max_tokens": 256,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("content", [{}])[0].get("text", "")
        except Exception as e:
            logger.warning(f"LLM request failed: {e}")
            return None

    def get_info(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "endpoint": settings.anthropic_base_url if self._available else None,
            "model": settings.anthropic_model if self._available else None,
        }


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
