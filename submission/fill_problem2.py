"""Problem 2 section of the submission document. Called from fill_submission.py (same python-docx objects, same helpers).

Every number is read from judge/results/*.json (produced by `python -m judge.experiments`) or from the audit log. Candidate-written text
(2.5 'explain your position-bias check') is NOT generated: it is read from submission/reflection_p2_posbias.txt if the candidate has saved it,
otherwise the box shows a factual hint list and a marker for the candidate to replace.
"""
import json
from pathlib import Path

from docx.shared import Inches  # noqa: F401  (screenshots can be dropped into the 2.5 boxes later)

DIAGRAM = """SUITE (JSON/YAML): { id, input, system_prompt, model_output | output_a+output_b, expected_output?, criteria? }
   |  load + validate (non-empty, unique ids, required fields)          python -m judge.run --suite S --out R
   v
JUDGE-PROMPT BUILDER   judge/rubric.py
   rubric (6 criteria x 1-5 anchors, weights) + length/authority rules + evidence rule + few-shot anchors   [variant = mitigated]
   mode: reference-based (expected_output given) | reference-free | pairwise A-vs-B
   |
   +--- pointwise ---------------------------+        +--- pairwise -----------------------------------------------+
   |                                         |        |  *** POSITION-BIAS ORDER SWAP (judge/core.py) ***          |
   |                                         |        |  call 1: (A,B)   call 2: (B,A)  -> map labels back         |
   |                                         |        |  agree -> winner | disagree -> TIE + flip flagged           |
   v                                         v        v                                                            |
JUDGE MODEL CALL  openai/gpt-oss-120b (OpenAI family) via Groq, temp 0   <== every call appended to judge_calls.jsonl
   |   (prompt messages + RAW response + tokens + latency + error)  = auditable and replayable (--replay)
   v
*** VERDICT PARSE + FALLBACK (judge/parsing.py) ***
   strip <think>/code fences -> outermost balanced JSON -> repair trailing commas/smart quotes -> schema check
   (every criterion present, score numeric 1..5, winner in A|B|TIE)
        |-- ok ------------------------------------------------------------------------------------------------>|
        '-- invalid -> ONE retry (corrective message, new cache tag) -> still invalid -> status 'unparseable'
                       (no score is invented; excluded from means, counted in the report)                       |
   v                                                                                                            |
GROUNDING CHECK: criterion >= 4 on correctness/faithfulness/completeness needs a verbatim evidence quote, else capped at 3
   v
AGGREGATION (judge/report.py): weighted overall -> pass rule (overall >= 3.5 AND no criterion <= 2)
   -> suite report: pass rate (+Wilson CI), mean/stdev, per-criterion means, unparseable count, tokens/calls
   -> A/B: both-order wins/ties, exact sign test -> winner / 'no significant difference' (+ sample-size caveat)
   v
report.json   +  judge/results/*.json (bias + validation experiments)   +   judge/logs/judge_calls.jsonl"""

PREREQ = ("Python 3.11+ (tested on 3.14), Windows 11 or Linux, no GPU, internet access to the Groq API (free tier is enough but slow: gpt-oss-120b 8,000 tokens/min; "
          "qwen3.8-27b only 1,000 output tokens/min, so the full experiment set takes about an hour; every call is cached so a re-run only pays for what is missing).")
ENVVARS = ("GROQ_API_KEY (secret), JUDGE_MODEL, GENERATOR_MODEL, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS, JUDGE_PASS_THRESHOLD, JUDGE_LOG, GROQ_BASE_URL "
           "(see .env.example; no value is committed).")
INSTALL = """git clone https://github.com/Mohityadav55199/SynergyLabs_Assignment2.git && cd SynergyLabs_Assignment2
python -m venv venv && source venv/bin/activate        # Windows: .\\venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env                                   # set GROQ_API_KEY (JUDGE_MODEL / GENERATOR_MODEL default to the two families below)
python -m pytest tests                                 # offline; no API calls"""
RUN_SUITE = """# pointwise suite -> structured report (per-criterion score + rationale + overall, pass rate, mean scores, tokens/calls)
python -m judge.run --suite judge/data/gold_cases.json --out judge/results/gold_report.json
# same suite with the un-mitigated 'baseline' judge prompt (for before/after comparisons)
python -m judge.run --suite judge/data/gold_cases.json --out judge/results/gold_report_baseline.json --variant baseline
# rebuild a report offline from the audit log (no API calls)
python -m judge.run --suite judge/data/gold_cases.json --out /tmp/replayed.json --replay"""
RUN_AB = """# generate outputs with the GENERATOR model under prompt v1 and prompt v2, then judge A-vs-B in both orders (+ pointwise per config)
python -m judge.generate --inputs judge/data/ab_inputs.json --out judge/data/ab_suite.json
python -m judge.run --suite judge/data/ab_suite.json --out judge/results/ab_report.json --pointwise
# every bias measurement + validation experiment (writes judge/results/*.json)
python -m judge.experiments all"""
INDEP = ("Yes. JUDGE_MODEL and GENERATOR_MODEL are separate environment variables (JudgeConfig in judge/config.py); the judge is used only in judge/core.py, "
         "the generator only in judge/generate.py. judge.run checks the two model families and refuses a same-family pair unless --allow-same-family is passed "
         "(then recorded in the report).")


def _load(J2):
    out = {}
    for f in J2.glob("*.json"):
        out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out
