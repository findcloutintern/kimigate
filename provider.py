import json
import uuid
import asyncio
import time
import logging
from typing import Any, AsyncIterator, Optional

import openai
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

from config import get_settings, NIM_BASE_URL, MODEL
from utils import (
    SSEBuilder,
    map_stop_reason,
    convert_messages,
    convert_tools,
    convert_system,
    ThinkParser,
    ContentType,
    ToolParser,
)
from utils.think_parser import extract_think_content

logger = logging.getLogger(__name__)


class RateLimiter:
    _instance: Optional["RateLimiter"] = None

    def __init__(self):
        if hasattr(self, "_init"):
            return
        settings = get_settings()
        self.limiter = AsyncLimiter(settings.rate_limit, settings.rate_window)
        self._blocked_until = 0.0
        self._init = True

    @classmethod
    def get(cls) -> "RateLimiter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def wait(self) -> bool:
        waited = False
        now = time.time()
        if now < self._blocked_until:
            wait_time = self._blocked_until - now
            logger.warning(f"rate limited, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            waited = True
        async with self.limiter:
            return waited

    def block(self, seconds: float = 60):
        self._blocked_until = time.time() + seconds


class NimProvider:
    def __init__(self):
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.nvidia_nim_api_key,
            base_url=NIM_BASE_URL,
            max_retries=2,
            timeout=300.0,
        )
        self._rate_limiter = RateLimiter.get()

    def _build_request(self, request: Any, stream: bool = False) -> dict:
        messages = convert_messages(request.messages)

        if request.system:
            sys_msg = convert_system(request.system)
            if sys_msg:
                messages.insert(0, sys_msg)

        body = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop_sequences:
            body["stop"] = request.stop_sequences
        if request.tools:
            body["tools"] = convert_tools(request.tools)

        # enable thinking mode for kimi k2.5
        body["extra_body"] = {
            "chat_template_kwargs": {
                "thinking": True
            }
        }

        return body

    async def stream(self, request: Any, input_tokens: int = 0) -> AsyncIterator[str]:
        waited = await self._rate_limiter.wait()

        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, MODEL, input_tokens)

        if waited:
            yield sse.message_start()
            for event in sse.emit_error("⏱️ rate limit active, resuming now..."):
                yield event

        body = self._build_request(request, stream=True)
        logger.info(f"stream: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}")

        if not waited:
            yield sse.message_start()

        think_parser = ThinkParser()
        tool_parser = ToolParser()
        finish_reason = None
        usage_info = None
        error_occurred = False

        try:
            stream = await self._client.chat.completions.create(**body, stream=True)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage_info = chunk.usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                # reasoning content
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    for event in sse.ensure_thinking():
                        yield event
                    yield sse.emit_thinking(reasoning)

                # text content
                if delta.content:
                    for part in think_parser.feed(delta.content):
                        if part.type == ContentType.THINKING:
                            for event in sse.ensure_thinking():
                                yield event
                            yield sse.emit_thinking(part.content)
                        else:
                            filtered, detected = tool_parser.feed(part.content)
                            if filtered:
                                for event in sse.ensure_text():
                                    yield event
                                yield sse.emit_text(filtered)

                            for tool in detected:
                                for event in sse.close_content():
                                    yield event
                                idx = sse.blocks.allocate_index()
                                yield sse.content_block_start(idx, "tool_use", id=tool["id"], name=tool["name"])
                                yield sse.content_block_delta(idx, "input_json_delta", json.dumps(tool["input"]))
                                yield sse.content_block_stop(idx)

                # native tool calls
                if delta.tool_calls:
                    for event in sse.close_content():
                        yield event
                    for tc in delta.tool_calls:
                        for event in self._process_tool_call(tc, sse):
                            yield event

        except Exception as e:
            logger.error(f"stream error: {type(e).__name__}: {e}")
            error_occurred = True
            for event in sse.close_content():
                yield event
            for event in sse.emit_error(str(self._map_error(e))):
                yield event

        # flush remaining
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                for event in sse.ensure_thinking():
                    yield event
                yield sse.emit_thinking(remaining.content)
            else:
                for event in sse.ensure_text():
                    yield event
                yield sse.emit_text(remaining.content)

        for tool in tool_parser.flush():
            for event in sse.close_content():
                yield event
            idx = sse.blocks.allocate_index()
            yield sse.content_block_start(idx, "tool_use", id=tool["id"], name=tool["name"])
            yield sse.content_block_delta(idx, "input_json_delta", json.dumps(tool["input"]))
            yield sse.content_block_stop(idx)

        # ensure at least one content block
        if not error_occurred and sse.blocks.text_index == -1 and not sse.blocks.tool_indices:
            for event in sse.ensure_text():
                yield event
            yield sse.emit_text(" ")

        for event in sse.close_all():
            yield event

        output_tokens = usage_info.completion_tokens if usage_info else sse.estimate_tokens()
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
        yield sse.done()

    async def complete(self, request: Any) -> dict:
        await self._rate_limiter.wait()
        body = self._build_request(request, stream=False)
        logger.info(f"complete: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}")

        try:
            response = await self._client.chat.completions.create(**body)
            return response.model_dump()
        except Exception as e:
            logger.error(f"complete error: {type(e).__name__}: {e}")
            raise self._map_error(e)

    def convert_response(self, response_json: dict, request: Any) -> dict:
        choice = response_json["choices"][0]
        message = choice["message"]
        content = []

        # extract reasoning
        reasoning = message.get("reasoning_content")
        if not reasoning:
            details = message.get("reasoning_details")
            if details and isinstance(details, list):
                reasoning = "\n".join(
                    item.get("text", "") for item in details if isinstance(item, dict)
                )

        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})

        # extract text
        if message.get("content"):
            raw = message["content"]
            if isinstance(raw, str):
                if not reasoning:
                    think, raw = extract_think_content(raw)
                    if think:
                        content.append({"type": "thinking", "thinking": think})
                if raw:
                    content.append({"type": "text", "text": raw})
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content.append(item)

        # extract tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = tc["function"].get("arguments", {})
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": args,
                })

        if not content:
            content.append({"type": "text", "text": " "})

        usage = response_json.get("usage", {})
        return {
            "id": response_json.get("id", f"msg_{uuid.uuid4()}"),
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": map_stop_reason(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    def _process_tool_call(self, tc, sse: SSEBuilder):
        tc_index = tc.index if tc.index is not None else len(sse.blocks.tool_indices)
        fn_delta = tc.function

        if fn_delta.name is not None:
            sse.blocks.tool_names[tc_index] = sse.blocks.tool_names.get(tc_index, "") + fn_delta.name

        if tc_index not in sse.blocks.tool_indices:
            name = sse.blocks.tool_names.get(tc_index, "")
            if name or tc.id:
                tool_id = tc.id or f"tool_{uuid.uuid4()}"
                yield sse.start_tool(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
        elif not sse.blocks.tool_started.get(tc_index) and sse.blocks.tool_names.get(tc_index):
            tool_id = tc.id or f"tool_{uuid.uuid4()}"
            name = sse.blocks.tool_names[tc_index]
            yield sse.start_tool(tc_index, tool_id, name)
            sse.blocks.tool_started[tc_index] = True

        args = fn_delta.arguments or ""
        if args:
            if not sse.blocks.tool_started.get(tc_index):
                tool_id = tc.id or f"tool_{uuid.uuid4()}"
                name = sse.blocks.tool_names.get(tc_index, "tool_call") or "tool_call"
                yield sse.start_tool(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
            yield sse.emit_tool(tc_index, args)

    def _map_error(self, e: Exception) -> Exception:
        if isinstance(e, openai.AuthenticationError):
            return Exception(f"authentication error: {e}")
        if isinstance(e, openai.RateLimitError):
            self._rate_limiter.block(60)
            return Exception(f"rate limit error: {e}")
        if isinstance(e, openai.BadRequestError):
            return Exception(f"bad request: {e}")
        if isinstance(e, openai.APIError):
            return Exception(f"api error: {e}")
        return e


_provider: Optional[NimProvider] = None


def get_provider() -> NimProvider:
    global _provider
    if _provider is None:
        _provider = NimProvider()
    return _provider


async def cleanup_provider():
    global _provider
    if _provider:
        client = getattr(_provider, "_client", None)
        if client and hasattr(client, "aclose"):
            await client.aclose()
    _provider = None
