import json
import uuid
import shlex
import logging
from typing import List, Optional, Union

import tiktoken
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from config import get_settings, Settings, MODEL
from models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse,
    Usage,
)
from provider import get_provider, NimProvider

logger = logging.getLogger(__name__)
router = APIRouter()
ENCODER = tiktoken.get_encoding("cl100k_base")


def _is_quota_check(request: MessagesRequest) -> bool:
    if request.max_tokens == 1 and len(request.messages) == 1:
        msg = request.messages[0]
        if msg.role == "user":
            content = msg.content
            if isinstance(content, str) and "quota" in content.lower():
                return True
            elif isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", "")
                    if text and "quota" in text.lower():
                        return True
    return False


def _is_title_generation(request: MessagesRequest) -> bool:
    if request.messages and request.messages[-1].role == "user":
        content = request.messages[-1].content
        target = "write a 5-10 word title"
        if isinstance(content, str) and target in content.lower():
            return True
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "text", "")
                if text and target in text.lower():
                    return True
    return False


def _is_suggestion_mode(request: MessagesRequest) -> bool:
    for msg in request.messages:
        if msg.role == "user":
            content = msg.content
            target = "[SUGGESTION MODE:"
            if isinstance(content, str) and target in content:
                return True
            elif isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", "")
                    if text and target in text:
                        return True
    return False


def _is_prefix_detection(request: MessagesRequest) -> tuple:
    if len(request.messages) != 1 or request.messages[0].role != "user":
        return False, ""

    content = ""
    msg = request.messages[0]
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        for block in msg.content:
            text = getattr(block, "text", "")
            if text:
                content += text

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except Exception:
            pass
    return False, ""


def _extract_prefix(command: str) -> str:
    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command)
        if not parts:
            return "none"

        cmd_start = 0
        for i, part in enumerate(parts):
            if "=" in part and not part.startswith("-"):
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first = cmd_parts[0]
        two_word = {"git", "npm", "docker", "kubectl", "cargo", "go", "pip", "yarn"}

        if first in two_word and len(cmd_parts) > 1:
            second = cmd_parts[1]
            if not second.startswith("-"):
                return f"{first} {second}"
            return first
        return first

    except ValueError:
        return command.split()[0] if command.split() else "none"


def _is_filepath_extraction(request: MessagesRequest) -> tuple:
    if len(request.messages) != 1 or request.messages[0].role != "user":
        return False, "", ""
    if request.tools:
        return False, "", ""

    content = ""
    msg = request.messages[0]
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        for block in msg.content:
            text = getattr(block, "text", "")
            if text:
                content += text

    if "Command:" not in content or "Output:" not in content:
        return False, "", ""
    if "filepaths" not in content.lower():
        return False, "", ""

    try:
        cmd_start = content.find("Command:") + len("Command:")
        output_marker = content.find("Output:", cmd_start)
        if output_marker == -1:
            return False, "", ""

        command = content[cmd_start:output_marker].strip()
        output = content[output_marker + len("Output:"):].strip()

        for marker in ["<", "\n\n"]:
            if marker in output:
                output = output.split(marker)[0].strip()

        return True, command, output
    except Exception:
        return False, "", ""


def _extract_filepaths(command: str, output: str) -> str:
    listing = {"ls", "dir", "find", "tree", "pwd", "cd", "mkdir", "rmdir", "rm"}
    reading = {"cat", "head", "tail", "less", "more", "bat", "type"}

    try:
        parts = shlex.split(command)
        if not parts:
            return "<filepaths>\n</filepaths>"

        base = parts[0].split("/")[-1].split("\\")[-1].lower()

        if base in listing:
            return "<filepaths>\n</filepaths>"

        if base in reading:
            paths = [p for p in parts[1:] if not p.startswith("-")]
            if paths:
                return f"<filepaths>\n{chr(10).join(paths)}\n</filepaths>"
            return "<filepaths>\n</filepaths>"

        return "<filepaths>\n</filepaths>"
    except Exception:
        return "<filepaths>\n</filepaths>"


def _count_tokens(messages: List, system: Optional[Union[str, List]] = None, tools: Optional[List] = None) -> int:
    total = 0

    if system:
        if isinstance(system, str):
            total += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total += len(ENCODER.encode(block.text))

    for msg in messages:
        if isinstance(msg.content, str):
            total += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                b_type = getattr(block, "type", None)
                if b_type == "text":
                    total += len(ENCODER.encode(getattr(block, "text", "")))
                elif b_type == "thinking":
                    total += len(ENCODER.encode(getattr(block, "thinking", "")))
                elif b_type == "tool_use":
                    total += len(ENCODER.encode(getattr(block, "name", "")))
                    total += len(ENCODER.encode(json.dumps(getattr(block, "input", {}))))
                    total += 10
                elif b_type == "tool_result":
                    content = getattr(block, "content", "")
                    if isinstance(content, str):
                        total += len(ENCODER.encode(content))
                    else:
                        total += len(ENCODER.encode(json.dumps(content)))
                    total += 5

    if tools:
        for tool in tools:
            tool_str = tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            total += len(ENCODER.encode(tool_str))

    total += len(messages) * 3
    if tools:
        total += len(tools) * 5

    return max(1, total)


@router.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    provider: NimProvider = Depends(get_provider),
    settings: Settings = Depends(get_settings),
):
    try:
        # optimization: fast prefix detection
        if settings.fast_prefix_detection:
            is_prefix, cmd = _is_prefix_detection(request)
            if is_prefix:
                return MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=MODEL,
                    content=[{"type": "text", "text": _extract_prefix(cmd)}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=5),
                )

        # optimization: mock quota check
        if settings.skip_quota_check and _is_quota_check(request):
            logger.info("skipped quota check")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4()}",
                model=MODEL,
                content=[{"type": "text", "text": "Quota check passed."}],
                stop_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
            )

        # optimization: skip title generation
        if settings.skip_title_generation and _is_title_generation(request):
            logger.info("skipped title generation")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4()}",
                model=MODEL,
                content=[{"type": "text", "text": "Conversation"}],
                stop_reason="end_turn",
                usage=Usage(input_tokens=100, output_tokens=5),
            )

        # optimization: skip suggestion mode
        if settings.skip_suggestion_mode and _is_suggestion_mode(request):
            logger.info("skipped suggestion mode")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4()}",
                model=MODEL,
                content=[{"type": "text", "text": ""}],
                stop_reason="end_turn",
                usage=Usage(input_tokens=100, output_tokens=1),
            )

        # optimization: mock filepath extraction
        if settings.skip_filepath_extraction:
            is_fp, cmd, output = _is_filepath_extraction(request)
            if is_fp:
                logger.info("mocked filepath extraction")
                return MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=MODEL,
                    content=[{"type": "text", "text": _extract_filepaths(cmd, output)}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=10),
                )

        if request.stream:
            input_tokens = _count_tokens(request.messages, request.system, request.tools)
            return StreamingResponse(
                provider.stream(request, input_tokens=input_tokens),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            response = await provider.complete(request)
            return provider.convert_response(response, request)

    except Exception as e:
        logger.error(f"error: {e}")
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=str(e))


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest):
    try:
        return TokenCountResponse(
            input_tokens=_count_tokens(request.messages, request.system, request.tools)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "provider": "nvidia_nim", "model": MODEL}


@router.get("/health")
async def health():
    return {"status": "healthy"}
