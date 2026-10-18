"""Groq chat client with retry/backoff, a JSONL audit log of every call (prompt + raw response), token/call tracking and replay.

Every call is appended to the log. The same log doubles as a cache: a call whose (model, temperature, messages, tag) was already
logged is answered from the log without touching the network, which makes runs resumable and the whole report replayable offline
(`replay_only=True`).
"""
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .config import JudgeConfig


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None
    cached: bool = False


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    errors: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["total_tokens"] = self.prompt_tokens + self.completion_tokens
        return d


def call_key(model: str, temperature: float, messages: List[Dict[str, str]], tag: str) -> str:
    blob = json.dumps({"m": model, "t": temperature, "msgs": messages, "tag": tag}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _extra_params(model: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}   # hidden reasoning tokens would eat the small output-token budget
    return {}


class LLMClient:
    def __init__(self, config: Optional[JudgeConfig] = None, log_path: Optional[Path] = None, replay_only: bool = False, max_attempts: int = 7, fail_fast_wait: Optional[float] = None):
        self.config = config or JudgeConfig()
        self.log_path = Path(log_path or self.config.log_path)
        self.replay_only = replay_only
        self.max_attempts = max_attempts
        self.fail_fast_wait = fail_fast_wait   # if set, a 429 whose retry-after exceeds this many seconds fails immediately (lets the caller fall back)
        self.usage: Dict[str, Usage] = {}
        self._cache: Dict[str, dict] = {}
        self._http = None
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cap = rec.get("max_tokens")
            truncated = bool(cap) and rec.get("completion_tokens", 0) >= cap   # hit the token cap: never reuse a cut-off answer
            if rec.get("error") is None and rec.get("key") and not truncated:
                self._cache.setdefault(rec["key"], rec)

    def _log(self, rec: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.0, max_tokens: int = 3000, tag: str = "", role: str = "judge") -> ChatResult:
        key = call_key(model, temperature, messages, tag)
        u = self.usage.setdefault(model, Usage())
        if key in self._cache:
            rec = self._cache[key]
            u.cached_calls += 1
            return ChatResult(rec["raw_response"], rec["prompt_tokens"], rec["completion_tokens"], rec["latency_s"], None, cached=True)
        if self.replay_only:
            return ChatResult("", error="replay_miss: this call is not in the log")
        if not self.config.api_key:
            return ChatResult("", error="GROQ_API_KEY is not set")

        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **_extra_params(model)}
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        text, pt, ct, err, t0 = "", 0, 0, None, time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self._http is None:
                    self._http = httpx.Client(timeout=120.0)
                r = self._http.post(f"{self.config.base_url}/chat/completions", json=payload, headers=headers)
                if r.status_code == 200:
                    body = r.json()
                    text = body["choices"][0]["message"].get("content") or ""
                    use = body.get("usage") or {}
                    pt, ct, err = use.get("prompt_tokens", 0), use.get("completion_tokens", 0), None
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = float(r.headers.get("retry-after") or min(2 ** attempt, 30))
                    if self.fail_fast_wait is not None and wait > self.fail_fast_wait:
                        err = f"rate_limited: retry-after {wait:.0f}s exceeds fail-fast limit ({err})"
                        break
                    u.retries += 1
                    print(f"  [llm] {model} {r.status_code}, retry {attempt}/{self.max_attempts} in {wait:.0f}s", file=sys.stderr)
                    time.sleep(wait + 0.5)
                    continue
                break  # 4xx other than 429: not retryable
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                err = f"{type(exc).__name__}: {exc}"
                u.retries += 1
                time.sleep(min(2 ** attempt, 20))
        latency = round(time.perf_counter() - t0, 3)
        u.calls += 1
        u.prompt_tokens += pt
        u.completion_tokens += ct
        if err:
            u.errors += 1
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "key": key, "tag": tag, "role": role, "model": model, "temperature": temperature,
               "max_tokens": max_tokens, "messages": messages, "raw_response": text, "prompt_tokens": pt, "completion_tokens": ct, "latency_s": latency, "error": err}
        self._log(rec)
        if err is None and not (max_tokens and ct >= max_tokens):
            self._cache[key] = rec
        return ChatResult(text, pt, ct, latency, err)

    def usage_summary(self) -> dict:
        return {m: u.as_dict() for m, u in self.usage.items()}
