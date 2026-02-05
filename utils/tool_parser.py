import re
import uuid
from enum import Enum
from typing import List, Dict, Any, Tuple


class ParserState(Enum):
    TEXT = 1
    MATCHING_FUNCTION = 2
    PARSING_PARAMETERS = 3


class ToolParser:
    """
    parses raw text tool calls in format:
    ● <function=Name><parameter=key>value</parameter>...
    """

    def __init__(self):
        self.state = ParserState.TEXT
        self.buffer = ""
        self.tool_id = None
        self.func_name = None
        self.params = {}
        self.func_pattern = re.compile(r"●\s*<function=([^>]+)>")
        self.param_pattern = re.compile(r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL)

    def feed(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        self.buffer += text
        detected = []
        filtered = ""

        while True:
            if self.state == ParserState.TEXT:
                if "●" in self.buffer:
                    idx = self.buffer.find("●")
                    filtered += self.buffer[:idx]
                    self.buffer = self.buffer[idx:]
                    self.state = ParserState.MATCHING_FUNCTION
                else:
                    filtered += self.buffer
                    self.buffer = ""
                    break

            if self.state == ParserState.MATCHING_FUNCTION:
                match = self.func_pattern.search(self.buffer)
                if match:
                    self.func_name = match.group(1).strip()
                    self.tool_id = f"toolu_{uuid.uuid4().hex[:8]}"
                    self.params = {}
                    self.buffer = self.buffer[match.end():]
                    self.state = ParserState.PARSING_PARAMETERS
                else:
                    if len(self.buffer) > 100:
                        filtered += self.buffer[0]
                        self.buffer = self.buffer[1:]
                        self.state = ParserState.TEXT
                    else:
                        break

            if self.state == ParserState.PARSING_PARAMETERS:
                finished = False

                while True:
                    param_match = self.param_pattern.search(self.buffer)
                    if param_match and "</parameter>" in param_match.group(0):
                        pre = self.buffer[:param_match.start()]
                        if pre:
                            filtered += pre
                        key = param_match.group(1).strip()
                        val = param_match.group(2).strip()
                        self.params[key] = val
                        self.buffer = self.buffer[param_match.end():]
                    else:
                        break

                if "●" in self.buffer:
                    idx = self.buffer.find("●")
                    if idx > 0:
                        filtered += self.buffer[:idx]
                        self.buffer = self.buffer[idx:]
                    finished = True
                elif len(self.buffer) > 0 and not self.buffer.lstrip().startswith("<"):
                    if "<parameter=" not in self.buffer:
                        filtered += self.buffer
                        self.buffer = ""
                        finished = True

                if finished:
                    detected.append({
                        "type": "tool_use",
                        "id": self.tool_id,
                        "name": self.func_name,
                        "input": self.params,
                    })
                    self.state = ParserState.TEXT
                else:
                    break

        return filtered, detected

    def flush(self) -> List[Dict[str, Any]]:
        detected = []
        if self.state == ParserState.PARSING_PARAMETERS:
            partial = re.finditer(r"<parameter=([^>]+)>(.*)$", self.buffer, re.DOTALL)
            for m in partial:
                self.params[m.group(1).strip()] = m.group(2).strip()

            detected.append({
                "type": "tool_use",
                "id": self.tool_id,
                "name": self.func_name,
                "input": self.params,
            })
            self.state = ParserState.TEXT
            self.buffer = ""
        return detected
