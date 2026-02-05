import re
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator
from enum import Enum


class ContentType(Enum):
    TEXT = "text"
    THINKING = "thinking"


@dataclass
class ContentChunk:
    type: ContentType
    content: str


class ThinkParser:
    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self._buffer = ""
        self._in_think = False

    def feed(self, content: str) -> Iterator[ContentChunk]:
        self._buffer += content

        while self._buffer:
            if not self._in_think:
                chunk = self._parse_outside()
                if chunk:
                    yield chunk
                else:
                    break
            else:
                chunk = self._parse_inside()
                if chunk:
                    yield chunk
                else:
                    break

    def _parse_outside(self) -> Optional[ContentChunk]:
        think_start = self._buffer.find(self.OPEN_TAG)
        orphan_close = self._buffer.find(self.CLOSE_TAG)

        # handle orphan </think>
        if orphan_close != -1 and (think_start == -1 or orphan_close < think_start):
            pre = self._buffer[:orphan_close]
            self._buffer = self._buffer[orphan_close + 8:]
            if pre:
                return ContentChunk(ContentType.TEXT, pre)
            return self._parse_outside()

        if think_start == -1:
            last_bracket = self._buffer.rfind("<")
            if last_bracket != -1:
                potential = self._buffer[last_bracket:]
                if len(potential) < 7 and self.OPEN_TAG.startswith(potential):
                    emit = self._buffer[:last_bracket]
                    self._buffer = self._buffer[last_bracket:]
                    if emit:
                        return ContentChunk(ContentType.TEXT, emit)
                    return None

            emit = self._buffer
            self._buffer = ""
            if emit:
                return ContentChunk(ContentType.TEXT, emit)
            return None
        else:
            pre = self._buffer[:think_start]
            self._buffer = self._buffer[think_start + 7:]
            self._in_think = True
            if pre:
                return ContentChunk(ContentType.TEXT, pre)
            return self._parse_inside()

    def _parse_inside(self) -> Optional[ContentChunk]:
        think_end = self._buffer.find(self.CLOSE_TAG)

        if think_end == -1:
            last_bracket = self._buffer.rfind("<")
            if last_bracket != -1 and len(self._buffer) - last_bracket < 8:
                potential = self._buffer[last_bracket:]
                if self.CLOSE_TAG.startswith(potential):
                    emit = self._buffer[:last_bracket]
                    self._buffer = self._buffer[last_bracket:]
                    if emit:
                        return ContentChunk(ContentType.THINKING, emit)
                    return None

            emit = self._buffer
            self._buffer = ""
            if emit:
                return ContentChunk(ContentType.THINKING, emit)
            return None
        else:
            thinking = self._buffer[:think_end]
            self._buffer = self._buffer[think_end + 8:]
            self._in_think = False
            if thinking:
                return ContentChunk(ContentType.THINKING, thinking)
            return self._parse_outside()

    def flush(self) -> Optional[ContentChunk]:
        if self._buffer:
            chunk_type = ContentType.THINKING if self._in_think else ContentType.TEXT
            content = self._buffer
            self._buffer = ""
            return ContentChunk(chunk_type, content)
        return None


def extract_think_content(text: str) -> Tuple[Optional[str], str]:
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        thinking = "\n".join(matches)
        remaining = pattern.sub("", text).strip()
        return thinking, remaining
    return None, text
