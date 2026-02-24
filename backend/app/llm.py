"""LLM integration for FinAlly chat actions."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from litellm import acompletion
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CHAT_MODEL = "openrouter/openai/gpt-oss-120b"
EXTRA_BODY = {"provider": {"order": ["Cerebras", "Groq"]}}
logger = logging.getLogger(__name__)


class TradeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)


class WatchlistAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    action: Literal["add", "remove"]


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    trades: list[TradeAction] = []
    watchlist_changes: list[WatchlistAction] = []


def _mock_response(user_message: str) -> ChatResponse:
    normalized = user_message.upper()
    trades: list[TradeAction] = []
    watchlist_changes: list[WatchlistAction] = []

    for side, quantity, ticker in re.findall(
        r"\b(BUY|SELL)\s+(\d+(?:\.\d+)?)\s+([A-Z]{1,10})\b", normalized
    ):
        trades.append(TradeAction(ticker=ticker, side=side.lower(), quantity=float(quantity)))

    for ticker in re.findall(r"\bADD\s+([A-Z]{1,10})\b", normalized):
        watchlist_changes.append(WatchlistAction(ticker=ticker, action="add"))

    for ticker in re.findall(r"\bREMOVE\s+([A-Z]{1,10})\b", normalized):
        watchlist_changes.append(WatchlistAction(ticker=ticker, action="remove"))

    msg = "Mock response: I prepared actions based on your request." if trades or watchlist_changes \
        else "Mock response: I reviewed your portfolio context."

    return ChatResponse(message=msg, trades=trades, watchlist_changes=watchlist_changes)


_FALLBACK = ChatResponse(message="I could not parse structured output from the model.")
_NO_KEY = ChatResponse(message="LLM is unavailable because OPENROUTER_API_KEY is not configured.")
_FAILED = ChatResponse(message="LLM request failed. Please verify your OPENROUTER_API_KEY and try again.")


async def generate_chat_response(
    user_message: str,
    history: list[dict],
    context: dict,
) -> ChatResponse:
    if os.environ.get("LLM_MOCK", "false").lower() == "true":
        return _mock_response(user_message)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return _NO_KEY

    context_json = json.dumps(context, separators=(",", ":"))
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are FinAlly, an AI trading assistant. Be concise and data-driven. "
                "Use the conversation history and the following account context to inform your responses. "
                "IMPORTANT: You can trade ANY valid US stock ticker. The backend resolves prices automatically — "
                "you do NOT need price data in the context to execute a trade. Never refuse a trade because "
                "a ticker is missing from the watchlist or because you don't see its price. "
                "Always include the trade in the trades array when the user asks to buy or sell. "
                "You may also add the ticker to their watchlist for tracking. "
                "Output strictly valid JSON using the required schema.\n\n"
                f"Account context:\n{context_json}"
            ),
        },
    ]

    for item in history:
        role = item.get("role", "user")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": str(item.get("content", ""))})

    messages.append({"role": "user", "content": user_message})

    try:
        response = await acompletion(
            model=CHAT_MODEL,
            api_key=api_key,
            messages=messages,
            response_format=ChatResponse,
            reasoning_effort="low",
            extra_body=EXTRA_BODY,
        )
    except Exception as exc:
        logger.warning("LiteLLM/OpenRouter request failed: %s", exc)
        return _FAILED

    try:
        content = response.choices[0].message.content or ""
        return ChatResponse.model_validate_json(content)
    except (ValidationError, ValueError, TypeError) as exc:
        logger.warning("Structured output parse failed: %s", exc)
        return _FALLBACK
