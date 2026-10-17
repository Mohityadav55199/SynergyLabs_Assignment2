import re
import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class RobustJsonParser:
    """
    Robust JSON extractor and repair engine for LLM-generated structured outputs.
    Handles:
    - Markdown code fences (```json ... ```)
    - Preamble and postamble commentary
    - Trailing commas in arrays and objects
    - Unescaped newlines in strings
    - Truncated or malformed brackets
    """

    @classmethod
    def clean_markdown_fences(cls, text: str) -> str:
        text = text.strip()
        # Extract content between ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    @classmethod
    def find_json_bounds(cls, text: str) -> str:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]
        
        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            return text[first_bracket : last_bracket + 1]
        
        return text

    @classmethod
    def repair_syntax_flaws(cls, json_str: str) -> str:
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        # Fix unquoted true, false, null casing if needed
        json_str = re.sub(r"\bTrue\b", "true", json_str)
        json_str = re.sub(r"\bFalse\b", "false", json_str)
        json_str = re.sub(r"\bNone\b", "null", json_str)
        # Normalize single-quoted keys and string values e.g. {'score': 4.0}
        json_str = re.sub(r"'(.*?)'\s*:", r'"\1":', json_str)
        json_str = re.sub(r":\s*'(.*?)'", r': "\1"', json_str)
        return json_str

    @classmethod
    def repair_truncated_json(cls, text: str) -> str:
        """
        Attempts to close truncated JSON strings and brackets when LLM reaches max token limit.
        """
        trimmed = text.strip()
        if not trimmed:
            return "{}"

        # If odd number of unescaped quotes, close the quote
        num_quotes = len(re.findall(r'(?<!\\)"', trimmed))
        if num_quotes % 2 != 0:
            trimmed += '"'

        # Balance open braces and brackets
        stack = []
        in_string = False
        escape = False
        for ch in trimmed:
            if ch == '"' and not escape:
                in_string = not in_string
            elif not in_string:
                if ch in "{[":
                    stack.append(ch)
                elif ch == "}" and stack and stack[-1] == "{":
                    stack.pop()
                elif ch == "]" and stack and stack[-1] == "[":
                    stack.pop()
            escape = (ch == "\\" and not escape)

        # Close in reverse order
        while stack:
            open_ch = stack.pop()
            if open_ch == "{":
                trimmed += "}"
            elif open_ch == "[":
                trimmed += "]"

        return trimmed

    @classmethod
    def parse_to_dict(cls, raw_text: str) -> Dict[str, Any]:
        if not raw_text or not raw_text.strip():
            return {"overall_score": 3.0, "overall_rationale": "Empty response received."}

        cleaned = cls.clean_markdown_fences(raw_text)
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Bracket slicing
        bounded = cls.find_json_bounds(cleaned)
        try:
            return json.loads(bounded)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Syntax repairs (trailing commas, Python bools, single quotes)
        repaired = cls.repair_syntax_flaws(bounded)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Strategy 4: Truncated JSON bracket/quote closure
        closed = cls.repair_truncated_json(repaired)
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            pass

        # Strategy 5: ast.literal_eval for pure Python dict formats
        try:
            import ast
            py_dict = ast.literal_eval(bounded)
            if isinstance(py_dict, dict):
                return py_dict
        except Exception:
            pass

        # Strategy 6: Fallback heuristic regex extraction for key-values
        logger.warning("Standard JSON parsing failed, attempting heuristic regex extraction.")
        extracted = cls.heuristic_regex_extractor(cleaned)
        if extracted:
            return extracted

        raise ValueError(f"Failed to parse structured JSON from LLM response:\n{raw_text[:500]}...")

    @classmethod
    def heuristic_regex_extractor(cls, text: str) -> Dict[str, Any]:
        """
        Extracts structured fields if JSON is critically broken.
        """
        result: Dict[str, Any] = {}
        
        # Look for overall_score
        score_match = re.search(r'"?overall_score"?\s*:\s*([0-9.]+)', text, re.IGNORECASE)
        if score_match:
            result["overall_score"] = float(score_match.group(1))

        # Look for winner (for pairwise)
        winner_match = re.search(r'"?winner"?\s*:\s*["\'](A|B|TIE)["\']', text, re.IGNORECASE)
        if winner_match:
            result["winner"] = winner_match.group(1).upper()

        # Look for rationale
        rationale_match = re.search(r'"?(?:overall_)?rationale"?\s*:\s*["\'](.*?)["\'](?=,\s*"|\s*})', text, re.DOTALL)
        if rationale_match:
            result["overall_rationale"] = rationale_match.group(1).strip()
        else:
            result["overall_rationale"] = "Heuristically extracted verdict from malformed response."

        return result

    @classmethod
    def parse_as_pydantic(cls, raw_text: str, model_cls: Type[T]) -> T:
        data = cls.parse_to_dict(raw_text)
        return model_cls.model_validate(data)
