import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Iterator

try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(openai_reason: Optional[str]) -> str:
    return STOP_REASON_MAP.get(openai_reason, "end_turn") if openai_reason else "end_turn"


@dataclass
class BlockManager:
    next_index: int = 0
    thinking_index: int = -1
    text_index: int = -1
    thinking_started: bool = False
    text_started: bool = False
    tool_indices: Dict[int, int] = field(default_factory=dict)
    tool_contents: Dict[int, str] = field(default_factory=dict)
    tool_names: Dict[int, str] = field(default_factory=dict)
    tool_started: Dict[int, bool] = field(default_factory=dict)

    def allocate_index(self) -> int:
        idx = self.next_index
        self.next_index += 1
        return idx


class SSEBuilder:
    def __init__(self, message_id: str, model: str, input_tokens: int = 0):
        self.message_id = message_id
        self.model = model
        self.input_tokens = input_tokens
        self.blocks = BlockManager()
        self._text = ""
        self._reasoning = ""

    def _event(self, event_type: str, data: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def message_start(self) -> str:
        return self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 1},
            },
        })

    def message_delta(self, stop_reason: str, output_tokens: int) -> str:
        return self._event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })

    def message_stop(self) -> str:
        return self._event("message_stop", {"type": "message_stop"})

    def done(self) -> str:
        return "[DONE]\n\n"

    def content_block_start(self, index: int, block_type: str, **kwargs) -> str:
        block: Dict[str, Any] = {"type": block_type}
        if block_type == "thinking":
            block["thinking"] = kwargs.get("thinking", "")
        elif block_type == "text":
            block["text"] = kwargs.get("text", "")
        elif block_type == "tool_use":
            block["id"] = kwargs.get("id", "")
            block["name"] = kwargs.get("name", "")
            block["input"] = kwargs.get("input", {})
        return self._event("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        })

    def content_block_delta(self, index: int, delta_type: str, content: str) -> str:
        delta: Dict[str, Any] = {"type": delta_type}
        if delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content
        return self._event("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        })

    def content_block_stop(self, index: int) -> str:
        return self._event("content_block_stop", {"type": "content_block_stop", "index": index})

    # thinking helpers
    def start_thinking(self) -> str:
        self.blocks.thinking_index = self.blocks.allocate_index()
        self.blocks.thinking_started = True
        return self.content_block_start(self.blocks.thinking_index, "thinking")

    def emit_thinking(self, content: str) -> str:
        self._reasoning += content
        return self.content_block_delta(self.blocks.thinking_index, "thinking_delta", content)

    def stop_thinking(self) -> str:
        self.blocks.thinking_started = False
        return self.content_block_stop(self.blocks.thinking_index)

    # text helpers
    def start_text(self) -> str:
        self.blocks.text_index = self.blocks.allocate_index()
        self.blocks.text_started = True
        return self.content_block_start(self.blocks.text_index, "text")

    def emit_text(self, content: str) -> str:
        self._text += content
        return self.content_block_delta(self.blocks.text_index, "text_delta", content)

    def stop_text(self) -> str:
        self.blocks.text_started = False
        return self.content_block_stop(self.blocks.text_index)

    # tool helpers
    def start_tool(self, tool_index: int, tool_id: str, name: str) -> str:
        block_idx = self.blocks.allocate_index()
        self.blocks.tool_indices[tool_index] = block_idx
        self.blocks.tool_contents[tool_index] = ""
        return self.content_block_start(block_idx, "tool_use", id=tool_id, name=name)

    def emit_tool(self, tool_index: int, partial_json: str) -> str:
        self.blocks.tool_contents[tool_index] += partial_json
        block_idx = self.blocks.tool_indices[tool_index]
        return self.content_block_delta(block_idx, "input_json_delta", partial_json)

    def stop_tool(self, tool_index: int) -> str:
        block_idx = self.blocks.tool_indices[tool_index]
        return self.content_block_stop(block_idx)

    # state helpers
    def ensure_thinking(self) -> Iterator[str]:
        if self.blocks.text_started:
            yield self.stop_text()
        if not self.blocks.thinking_started:
            yield self.start_thinking()

    def ensure_text(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking()
        if not self.blocks.text_started:
            yield self.start_text()

    def close_content(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking()
        if self.blocks.text_started:
            yield self.stop_text()

    def close_all(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking()
        if self.blocks.text_started:
            yield self.stop_text()
        for tool_index in list(self.blocks.tool_indices.keys()):
            yield self.stop_tool(tool_index)

    def emit_error(self, message: str) -> Iterator[str]:
        idx = self.blocks.allocate_index()
        yield self.content_block_start(idx, "text")
        yield self.content_block_delta(idx, "text_delta", message)
        yield self.content_block_stop(idx)

    def estimate_tokens(self) -> int:
        if ENCODER:
            text_tokens = len(ENCODER.encode(self._text))
            reasoning_tokens = len(ENCODER.encode(self._reasoning))
            tool_tokens = sum(
                len(ENCODER.encode(self.blocks.tool_names.get(i, ""))) +
                len(ENCODER.encode(c)) + 10
                for i, c in self.blocks.tool_contents.items()
            )
            return text_tokens + reasoning_tokens + tool_tokens
        return len(self._text) // 4 + len(self._reasoning) // 4 + len(self.blocks.tool_indices) * 50
