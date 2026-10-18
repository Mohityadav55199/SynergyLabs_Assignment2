"""Fills Problems 1 AND 2 of submission_template.docx -> Submission_Problem1_and_2.docx (one run, always from the pristine template).

  pip install python-docx
  python submission/fill_submission.py

Problem 1 numbers come from submission/problem1_results/*.json (copied from the Problem 1 repo, SynergyLabs_Assignment);
Problem 2 numbers come from judge/results/*.json and judge/logs/judge_calls.jsonl, so re-running after
`python -m judge.experiments all --force` updates them.
Candidate-written text is read from submission/reflection_*.txt and is never generated here:
  reflection_walkthrough.txt, reflection_broke.txt (Problem 1), reflection_ai_disclosure_p1.txt (Problem 1 AI disclosure, hand-edited),
  reflection_p2_posbias.txt (Problem 2.5 "explain your position-bias check", own words, NOT written by the AI).
"""
import json
from pathlib import Path

import docx
from docx.shared import Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "submission" / "submission_template.docx"
OUT = ROOT / "submission" / "Submission_Problem1_and_2.docx"
P1 = ROOT / "submission" / "problem1_results"
J2 = ROOT / "judge" / "results"

res = json.loads((P1 / "results_lancedb.json").read_text(encoding="utf-8"))
qd = json.loads((P1 / "results_qdrant.json").read_text(encoding="utf-8"))
scale = json.loads((P1 / "results_scale.json").read_text(encoding="utf-8"))
ret = res["retrieval_evaluation"]["aggregate"]["unfiltered"]
lat = res["retrieval_evaluation"]["latency_ms"]
ans = res["answer_evaluation"]
ok = ans["status"] == "OK"
PEND = "n/a"
P2_MODELS_TEXT = "judge: openai/gpt-oss-120b (OpenAI family; gpt-oss-20b, same family, for the sycophancy / A-B / retest runs after the 120b daily quota ran out) · generator: qwen/qwen3.8-27b (Qwen family)"

d = docx.Document(SRC)
body = [el for el in d.element.body.iterchildren() if el.tag.endswith("}p")]
T = d.tables


def set_cell(cell, text, mono=False, size=9):
    paras = cell.paragraphs
    for p in paras[1:]:
        p._element.getparent().remove(p._element)
    p = paras[0]
    for r in p.runs[1:]:
        r._element.getparent().remove(r._element)
    run = p.runs[0] if p.runs else p.add_run()
    run.text = text
    if mono:
        run.font.name = "Consolas"
    run.font.size = Pt(size)


def set_after_label(idx, text):
    runs = Paragraph(body[idx], d).runs
    runs[-1].text = text


# ---- candidate details ----
rows = T[0].rows
for i, v in enumerate(["Mohit Yadav", "mohityadav55199@gmail.com", "2026-10-18",
                       "Problem 1: https://github.com/Mohityadav55199/SynergyLabs_Assignment · Problem 2: https://github.com/Mohityadav55199/SynergyLabs_Assignment2",
                       "[ candidate: enter the combined total for Problems 1 + 2 — Problem 1 alone was ~8 hours ]", "LanceDB (primary), embedded Qdrant benchmarked as a second store",
                       P2_MODELS_TEXT], start=1):
    set_cell(rows[i].cells[1], v)

# ---- 1.1 diagram ----
set_cell(T[1].rows[0].cells[0], """INGESTION (idempotent)
  PDF / HTML / MD --> parse (pypdf / BeautifulSoup / text)
     --> chunk: recursive, size 500 chars, overlap 50 (configurable)
     --> fingerprint = SHA-256(doc_id, chunk_index, text)
     --> already stored? --yes--> SKIP (no duplicate vector)
                         --no---> embed: BAAI/bge-small-en-v1.5, 384-d, local ONNX
     --> LanceDB (primary) / Qdrant embedded: vector + text + metadata {format, doc_id, filename, section}

QUERY
  POST /query or CLI --> embed query --> top-k cosine search (+ metadata filter on format|doc_id|filename|section)
     --> best score >= SIMILARITY_THRESHOLD (0.68)?
           NO  --> *** NO RELEVANT CONTEXT branch ***: fixed refusal, LLM NOT called, 0 tokens, no citations
           YES --> grounded prompt (numbered chunks) --> Groq qwen/qwen3.8-27b, temp 0
                   --> answer with [Chunk-N] citations (only cited chunks returned)
                       + per-query log: latency (embed/search/gen/total), chunk count, token usage
                   (LLM down --> labelled extractive fallback, never scored as an LLM answer)
(Mermaid version of this diagram: README section 1)""", mono=True, size=8)

# ---- 1.2 setup ----
set_after_label(18, "Python 3.11+ (tested on 3.14), Windows 11 / Linux, no GPU, ~1 GB disk, internet for the first embedding-model download and for the Groq API.")
set_after_label(19, "GROQ_API_KEY, GROQ_MODEL, JUDGE_MODEL, DEFAULT_VECTOR_STORE, LANCEDB_DIR, QDRANT_DIR, EMBEDDING_MODEL, EMBEDDING_DIM, "
                    "DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MAX_UPLOAD_MB, CORPUS_DIR, SIMILARITY_THRESHOLD, HOST, PORT, CORS_ORIGINS, ADMIN_TOKEN (see .env.example).")
set_cell(T[2].rows[0].cells[0], """git clone https://github.com/Mohityadav55199/SynergyLabs_Assignment.git && cd SynergyLabs_Assignment
python -m venv venv && source venv/bin/activate      # Windows: .\\venv\\Scripts\\activate
pip install -r requirements-dev.txt
cp .env.example .env                                  # set GROQ_API_KEY
python -m pytest tests                                # 55 tests""", mono=True)
set_cell(T[3].rows[0].cells[0], """# defaults: chunk size 500 chars, overlap 50 chars
python -m src.cli ingest --store lancedb              # ingests data/corpus (PDF, HTML, MD)
python -m src.cli ingest --store lancedb --chunk-size 400 --overlap 40
python -m src.cli ingest --store lancedb              # run again: Newly inserted 0, Duplicates skipped 100""", mono=True)
set_cell(T[4].rows[0].cells[0], """python -m src.cli query "What is the memory overhead per vector for an HNSW index?" --k 4 --filter format=pdf
python -m src.cli serve   # then:
curl -s localhost:8000/query -H 'content-type: application/json' \\
  -d '{"query":"What is HNSW and what recall does it provide?","top_k":4,"metadata_filter":{"format":"pdf"}}'
# live: https://cost-efficient-rag-service.onrender.com/docs""", mono=True)
set_after_label(26, "LanceDB: an embedded library over files, so zero idle compute and no server to pay for, with metadata filtering and IVF_PQ for scale.")

# ---- 1.3.1 retrieval ----
set_after_label(31, "k = 4 (25 answerable questions of 30; 6 documents / 100 chunks; unfiltered search)")
rt = T[5].rows
set_cell(rt[1].cells[1], f"Hit Rate {ret['hit_rate']} / Recall {ret['recall']}")
set_cell(rt[1].cells[2], "Hit = any relevant chunk in top 4 (23/25). Recall = relevant retrieved / ALL relevant chunks in the corpus. Misses: q10, q19.")
set_cell(rt[2].cells[1], str(ret["mrr"]))
set_cell(rt[2].cells[2], "mean of 1/rank of first relevant chunk (0 if none in top 4)")
set_cell(rt[3].cells[1], str(ret["ndcg"]))
set_cell(rt[3].cells[2], "binary gain; ideal DCG places all relevant chunks (up to k) first, so <= 1 (the original harness reported 1.64: a bug, fixed)")
set_cell(rt[4].cells[1], str(ret["context_precision"]))
set_cell(rt[4].cells[2], "Relevant chunk = from the target file AND contains the question's hand-picked evidence phrase, computed over the whole corpus with the ingestion chunker. "
                         "Context precision is rank-aware (RAGAS definition). Qdrant gives identical numbers (both exact cosine search).")

# ---- 1.3.2 answer quality ----
at = T[6].rows
if ok:
    vals = [(ans["faithfulness_mean"], f"claim-level, judge {ans['judge_model']}"), (ans["answer_relevance_mean"], "1-5 rubric rescaled to 0-1"),
            (ans["exact_match_mean"], "normalised string match vs gold"), (ans["token_f1_mean"], "token F1 vs gold")]
    for i, (v, m) in enumerate(vals, start=1):
        set_cell(at[i].cells[1], str(v))
        set_cell(at[i].cells[2], "LLM-judge" if i <= 2 else "gold answers")
        set_cell(at[i].cells[3], m + (" (24 answered of 25; refusals 5/5 unanswerable, 1/25 false refusal)" if i == 1 else ""))
else:
    for i in range(1, 5):
        set_cell(at[i].cells[1], "PENDING")
        set_cell(at[i].cells[3], PEND)

# ---- 1.3.3 cost ----
ct = T[7].rows
cost = [("$0.04 marginal on an existing host ($0.01 S3-only); $24.57 on a dedicated t4g.medium VM", "$50.00 (Pinecone Standard minimum); Qdrant Cloud free 1 GB tier = $0",
         "saving is ~$50 vs the Pinecone minimum, or ~$25 if a VM must be provisioned"),
        ("$0.40 marginal ($0.08 S3-only); $49.46 dedicated t4g.large", "$50.00 (Pinecone Standard minimum); Qdrant Cloud: no public rate card",
         "with a dedicated VM the saving is ~$0.5; marginal saving ~$50"),
        ("$3.96 marginal ($0.76 S3-only); $53.02 dedicated VM", "$50.00 (Pinecone minimum; storage+reads ~$14 stay under it)",
         "dedicated VM costs MORE than Pinecone's minimum at 10M; only the shared-host case is cheaper")]
for i, (a, b, c) in enumerate(cost, start=1):
    set_cell(ct[i].cells[1], a)
    set_cell(ct[i].cells[2], b)
    set_cell(ct[i].cells[3], c)
set_cell(T[8].rows[0].cells[0],
         "384-d float32 (1,536 B) + ~1 KB chunk text/metadata = 2.5 KB/vector; x1.3 index overhead. ~15,000 queries/month (~500/day). "
         "Pinecone Standard: $50/month minimum, $0.33/GB-month storage, $16-18 per M read units, $4-4.5 per M write units (pinecone.io/pricing, retrieved 2026-10-18; "
         "read-unit sizing from Pinecone docs, unverified). EC2 t4g.medium $24.53 / t4g.large $49.06 per month (third-party aggregator, us-east-1 on-demand). "
         "EBS gp3 $0.08 and S3 Standard $0.023 per GB-month, 1.5x disk headroom. Embeddings are local (free). LLM cost is identical on both sides and excluded. "
         "Single node, no replication. Excludes engineer time. Full model with sources: evaluation/cost_analysis.md (python -m evaluation.cost_model).")

# ---- 1.3.4 latency ----
lt = T[9].rows
l100 = scale["lancedb_100000"]
set_cell(lt[1].cells[1], f"{lat['vector_search_p50']} (100 chunks) / {l100['ivf_pq']['p50_ms']} (100K, IVF_PQ) / {l100['flat_scan']['p50_ms']} (100K, flat)")
set_cell(lt[1].cells[2], f"LanceDB, k=4, warm, 150 samples (100 chunks); 100K row: synthetic vectors, 100 queries, laptop CPU. Query embedding is separate: p50 {lat['query_embedding_p50']} ms.")
set_cell(lt[2].cells[1], f"{lat['vector_search_p95']} (100 chunks) / {l100['ivf_pq']['p95_ms']} (100K, IVF_PQ) / {l100['flat_scan']['p95_ms']} (100K, flat)")
set_cell(lt[2].cells[2], f"Query embedding p95 {lat['query_embedding_p95']} ms. Qdrant local on 100 chunks: p50 {qd['retrieval_evaluation']['latency_ms']['vector_search_p50']}, p95 {qd['retrieval_evaluation']['latency_ms']['vector_search_p95']} ms.")
if ok:
    set_cell(lt[3].cells[1], str(ans["end_to_end_latency_ms"]["p95"]))
    set_cell(lt[3].cells[2], "includes Groq generation, 30 questions")
else:
    set_cell(lt[3].cells[1], "PENDING")
    set_cell(lt[3].cells[2], PEND)

# ---- 1.4 design decisions ----
design = [
    "The brief's cost problem is always-on pods. LanceDB is a library over files, so an idle index costs only disk, and it supports metadata filters and IVF_PQ. "
    "pgvector needs a running Postgres (the always-on cost we avoid, though it wins if Postgres already exists); FAISS has no filtering/persistence; Chroma works but I wanted control over the on-disk "
    "format; sqlite-vec is brute-force only. Embedded Qdrant is benchmarked as the second store: identical retrieval (both exact cosine), faster on 100 chunks, but its local mode is brute-force numpy, not HNSW.",
    "Default 500 chars / 50 overlap, recursive splitter (paragraph, line, sentence, word) so chunks end on sentence boundaries. Measured (README 6.3): 200/20 and 500/0 are worse (hit 0.88, MRR 0.72-0.75); "
    "350-800 are on par; 1200/120 scored higher (hit 1.00, MRR 0.92). I did not adopt it because my relevance labels reward large chunks, k=4 x 1200 chars is ~4x the prompt tokens, and the corpus is tiny. "
    "The default is a defensible middle, not a tuned optimum; it should be re-tuned on the real corpus with the answer layer.",
    "BAAI/bge-small-en-v1.5, 384-d, run locally via ONNX: $0 per embedding, ~40 ms/query on CPU, 1.5 KB/vector. Larger models (768-1024-d) cost 2-3x the storage/RAM and CPU for gains I did not measure; "
    "a hosted embedding API would add per-token cost and a network hop. Weakness seen: generic RAG vocabulary confuses it (q10).",
    "Two layers. (1) Similarity gate before the LLM: if the best chunk's cosine score is < 0.68 the service returns a fixed refusal, calls no LLM and uses 0 tokens. Tuned on the eval set: answerable top-1 >= 0.726, "
    "unanswerable <= 0.713, so it blocks 4/5 unanswerable questions; the near-domain one (0.713) is not separable. (2) The prompt tells the model to answer only from numbered chunks, cite [Chunk-N], and otherwise emit the exact refusal sentence, at temperature 0; "
    "refusals carry no citations. If the LLM is down, a labelled verbatim passage is returned, never a made-up answer.",
    "Each chunk is fingerprinted with SHA-256(doc_id, chunk_index, text), where doc_id is derived from the file's content hash. Before embedding, existing hashes are read from the store and matching chunks are skipped, "
    "so a re-run embeds nothing and inserts nothing (100 chunks -> 100 stored -> still 100). A hash-read failure raises instead of returning an empty set (which would silently re-insert everything). "
    "An edited file gets a new doc_id and its old vectors are deleted, so stale text cannot be retrieved. Writes are serialised with a lock. Tested on both stores.",
    "Accepted: single node and no replication; one writer at a time (file lock); an ANN index is mandatory past ~50K vectors (flat scan was 598 ms p50 at 100K); I own ops; local Qdrant is brute-force. "
    "I would switch back to managed when (a) sustained QPS/p99 needs a replicated in-memory index, (b) several services must write to one index, (c) a contractual availability SLA is required, or "
    "(d) the index is small and nobody owns ops, since Pinecone's $50 minimum (or Qdrant's free tier) is then cheaper than the engineering time.",
]
for t_idx, text in zip(range(10, 16), design):
    set_cell(T[t_idx].rows[0].cells[0], text, size=10)

# ---- 1.5 evidence ----
from docx.shared import Inches
shot = ROOT / "submission" / "screenshots" / "end_to_end_query.png"
if shot.exists():
    cell = T[16].rows[0].cells[0]
    set_cell(cell, "Live deployment (https://synergylabs-lilac.vercel.app), query: 'What is HNSW and what recall accuracy does it provide?'. "
                   "Shown: cited answer ([Chunk-1], [Chunk-2]), 800 ms retrieval, 1378.64 ms total, 525 tokens, model qwen/qwen3.8-27b, 3 chunks used; "
                   "sidebar shows 6 documents / 100 chunks.", size=9)
    cell.add_paragraph().add_run().add_picture(str(shot), width=Inches(6.2))

set_cell(T[17].rows[0].cells[0], """$ python -m src.cli ingest --store lancedb      (first run, empty index)
Total chunks generated: 100 | Newly inserted chunks: 100 | Duplicates skipped: 0
$ python -m src.cli ingest --store lancedb      (second run, same corpus)
Total chunks generated: 100 | Newly inserted chunks: 0   | Duplicates skipped: 100 (Idempotent)
$ python -m src.cli stats --store lancedb
  total_chunks: 100
(Same on --store qdrant. Also asserted in tests/test_stores.py and tests/test_api.py.)""", mono=True)

# ---- 1.5 candidate-written reflection (text supplied by the candidate in submission/reflection_walkthrough.txt) ----
walk = ROOT / "submission" / "reflection_walkthrough.txt"
if walk.exists():
    set_cell(T[18].rows[0].cells[0], walk.read_text(encoding="utf-8").strip(), size=10)

broke = ROOT / "submission" / "reflection_broke.txt"
if broke.exists():
    set_cell(T[19].rows[0].cells[0], broke.read_text(encoding="utf-8").strip(), size=10)

# ---- 1.5 AI usage disclosure (the candidate's hand-edited text, preserved in reflection_ai_disclosure_p1.txt) ----
disc = ROOT / "submission" / "reflection_ai_disclosure_p1.txt"
if disc.exists():
    set_cell(T[20].rows[0].cells[0], disc.read_text(encoding="utf-8").strip(), size=10)

# ---- Problem 2 ----
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'submission'))
from p2_fill_body import fill_problem2
fill_problem2(d, body, T, set_cell, set_after_label, ROOT)
try:
    d.save(OUT)
except PermissionError:  # the target is open in Word
    OUT = OUT.with_name("Submission_Problem1_new.docx")
    d.save(OUT)
print("wrote", OUT, "| P1 answer layer:", ans["status"], "| P2 filled from judge/results")
