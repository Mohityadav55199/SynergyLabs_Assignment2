"""Robust extraction + schema validation of judge output. Never raises, never invents a score."""
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .rubric import CRITERIA_IDS


@dataclass
class Parsed:
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    notes: Optional[List[str]] = None


def strip_noise(raw: str) -> str:
    """Drop <think>/<thinking> blocks (also an unclosed one) and markdown code fences."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw or "", flags=re.S | re.I)
    text = re.sub(r"<think(?:ing)?>.*\Z", "", text, flags=re.S | re.I)
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S)
    if fence and "{" in fence.group(1):
        text = fence.group(1)
    return text.strip()


def _balanced_objects(text: str) -> List[str]:
    """All top-level balanced {...} spans (string-aware), outermost first by position."""
    spans, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    return spans


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)                 # trailing commas
    s = s.replace("“", '"').replace("”", '"')   # smart quotes
    return s


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Return the first parseable JSON object found in the model text (largest candidate first), else None."""
    text = strip_noise(raw)
    if not text:
        return None
    candidates = sorted(_balanced_objects(text), key=len, reverse=True)
    if not candidates:
        a, b = text.find("{"), text.rfind("}")
        candidates = [text[a:b + 1]] if a != -1 and b > a else []
    for cand in candidates:
        for attempt in (cand, _repair(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def parse_pointwise(raw: str, expected_ids: Optional[List[str]] = None) -> Parsed:
    expected_ids = expected_ids or CRITERIA_IDS
    obj = extract_json(raw)
    if obj is None:
        return Parsed(False, error="no_json_found" if (raw or "").strip() else "empty_response")
    crit = obj.get("criteria")
    if isinstance(crit, list):  # tolerate [{"id": ..., "score": ...}]
        crit = {c.get("id") or c.get("criterion"): c for c in crit if isinstance(c, dict)}
    if not isinstance(crit, dict):
        return Parsed(False, error="schema: 'criteria' missing or not an object")
    out, notes = {}, []
    for cid in expected_ids:
        c = crit.get(cid)
        if not isinstance(c, dict) or "score" not in c:
            return Parsed(False, error=f"schema: criterion '{cid}' missing or has no score")
        s = _num(c["score"])
        if c["score"] is None:
            out[cid] = {"score": None, "rationale": str(c.get("rationale", "")), "evidence": None}
            continue
        if s is None or not (1 <= s <= 5):
            return Parsed(False, error=f"schema: criterion '{cid}' score {c['score']!r} not a number in 1..5")
        ev = c.get("evidence")
        out[cid] = {"score": s, "rationale": str(c.get("rationale", "")).strip(), "evidence": (str(ev).strip() if ev not in (None, "") else None)}
    ov = obj.get("overall")
    overall_reported, overall_rationale = None, ""
    if isinstance(ov, dict):
        overall_reported, overall_rationale = _num(ov.get("score")), str(ov.get("rationale", "")).strip()
    elif ov is not None:
        overall_reported = _num(ov)
    if overall_reported is not None and not (1 <= overall_reported <= 5):
        notes.append(f"overall {overall_reported} out of range; ignored")
        overall_reported = None
    return Parsed(True, {"criteria": out, "overall_reported": overall_reported, "overall_rationale": overall_rationale}, notes=notes)


def parse_pairwise(raw: str) -> Parsed:
    obj = extract_json(raw)
    if obj is None:
        return Parsed(False, error="no_json_found" if (raw or "").strip() else "empty_response")
    w = str(obj.get("winner", "")).strip().upper()
    w = {"RESPONSE A": "A", "RESPONSE B": "B", "DRAW": "TIE", "EQUAL": "TIE", "NEITHER": "TIE"}.get(w, w)
    if w not in ("A", "B", "TIE"):
        return Parsed(False, error=f"schema: winner {obj.get('winner')!r} not in A|B|TIE")
    conf = str(obj.get("confidence", "")).strip().lower()
    return Parsed(True, {"winner": w, "confidence": conf if conf in ("high", "medium", "low") else None, "rationale": str(obj.get("rationale", "")).strip()})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'`‘’“”]", "", s or "")).strip().lower()


def evidence_supported(evidence: Optional[str], model_output: str) -> bool:
    """A quoted `evidence` counts only if it is (after whitespace/quote/case normalisation) a verbatim fragment of the output.
    '...' separates fragments; every fragment of 3+ chars must appear."""
    if not evidence:
        return False
    out = _norm(model_output)
    frags = [f for f in (_norm(p) for p in re.split(r"\.\.\.|…", evidence)) if len(f) >= 3]
    return bool(frags) and all(f in out for f in frags)
