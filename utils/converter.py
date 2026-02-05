import json
from typing import Any, Dict, List, Optional


def _get_attr(block: Any, attr: str, default: Any = None) -> Any:
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def _get_type(block: Any) -> Optional[str]:
    return _get_attr(block, "type")


def convert_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    result = []
    for msg in messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            if role == "assistant":
                result.extend(_convert_assistant(content))
            elif role == "user":
                result.extend(_convert_user(content))
        else:
            result.append({"role": role, "content": str(content)})
    return result


def _convert_assistant(content: List[Any]) -> List[Dict[str, Any]]:
    text_parts = []
    tool_calls = []
    reasoning_parts = []

    for block in content:
        block_type = _get_type(block)

        if block_type == "text":
            text_parts.append(_get_attr(block, "text", ""))
        elif block_type == "thinking":
            reasoning_parts.append(_get_attr(block, "thinking", ""))
        elif block_type == "tool_use":
            tool_input = _get_attr(block, "input", {})
            tool_calls.append({
                "id": _get_attr(block, "id"),
                "type": "function",
                "function": {
                    "name": _get_attr(block, "name"),
                    "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                },
            })

    actual_content = []
    if reasoning_parts:
        reasoning_str = "\n".join(reasoning_parts)
        actual_content.append(f"<think>\n{reasoning_str}\n</think>")
    if text_parts:
        actual_content.append("\n".join(text_parts))

    content_str = "\n\n".join(actual_content)
    if not content_str and not tool_calls:
        content_str = " "

    msg: Dict[str, Any] = {"role": "assistant", "content": content_str}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _convert_user(content: List[Any]) -> List[Dict[str, Any]]:
    result = []
    text_parts = []

    for block in content:
        block_type = _get_type(block)

        if block_type == "text":
            text_parts.append(_get_attr(block, "text", ""))
        elif block_type == "tool_result":
            tool_content = _get_attr(block, "content", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in tool_content
                )
            result.append({
                "role": "tool",
                "tool_call_id": _get_attr(block, "tool_use_id"),
                "content": str(tool_content) if tool_content else "",
            })

    if text_parts:
        result.append({"role": "user", "content": "\n".join(text_parts)})
    return result


def convert_tools(tools: List[Any]) -> List[Dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    } for tool in tools]


def convert_system(system: Any) -> Optional[Dict[str, str]]:
    if isinstance(system, str):
        return {"role": "system", "content": system}
    elif isinstance(system, list):
        text_parts = []
        for block in system:
            if _get_type(block) == "text":
                text_parts.append(_get_attr(block, "text", ""))
        if text_parts:
            return {"role": "system", "content": "\n\n".join(text_parts).strip()}
    return None
