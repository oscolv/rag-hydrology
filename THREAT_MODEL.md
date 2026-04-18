# Threat Model — RAG Pipeline

This document captures the security posture of the pipeline: what we protect,
who we protect it from, and where the mitigations live in code. It is the
backing document for [`SECURITY.md`](./SECURITY.md).

The model uses a lightweight STRIDE framing (Spoofing, Tampering, Repudiation,
Information disclosure, Denial-of-service, Elevation-of-privilege). Each
threat is paired with the concrete mitigation and the file/line where it is
enforced, so a reviewer can verify "is this still true?" by reading code.

---

## 1. System overview

```
┌──────────┐   ingest   ┌─────────────┐   retrieve   ┌──────────────┐
│   PDFs   │  ────────▶ │   ChromaDB  │  ──────────▶ │   Generation │ ─▶ Answer
│ (disk)   │            │  + bm25.pkl │              │   (LLM call) │
└──────────┘            └─────────────┘              └──────────────┘
                                                            ▲
                                                            │
                                                      User CLI input
```

Components:

- **Ingestion** (`src/rag/ingest.py`) — parses PDFs, chunks, embeds, writes
  Chroma + `bm25.pkl`.
- **Retrieval** (`src/rag/retrieval.py`) — hybrid dense/BM25, reranking.
- **Generation** (`src/rag/generation.py`) — prompt construction, LLM call,
  Self-RAG grading.
- **CLI** (`src/rag/cli.py`) — Typer entrypoint, interactive chat.
- **External services** — OpenAI / OpenRouter / Cohere (rerank) / HF
  (embeddings, optional).

## 2. Trust boundaries

| # | Boundary | Trusted side | Untrusted side |
|---|----------|--------------|----------------|
| B1 | CLI process ↔ local user | process env, config | CLI arguments, chat input |
| B2 | Ingestion ↔ PDF corpus | pipeline code | PDF content (could contain injected prompts) |
| B3 | Pipeline ↔ LLM provider | request payload | LLM response text (could be malformed JSON, prose, attempted instruction injection) |
| B4 | Pipeline ↔ on-disk index | running process | `bm25.pkl`, Chroma sqlite (could be swapped by attacker with disk access) |
| B5 | Pipeline ↔ logs/stderr | structured events | anything a future reader might see (secrets must not land here) |

## 3. Assets

| Asset | Why it matters |
|-------|----------------|
| API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `COHERE_API_KEY`) | Direct billing/abuse if leaked |
| User questions | May contain private research context |
| PDF corpus | May be proprietary or embargoed |
| Answer integrity | A tampered pipeline could return plausible-but-wrong medical/scientific claims |
| Developer environment | A pickle-RCE would compromise the user's home directory |

## 4. Threats and mitigations

### T1 — Prompt-template injection via PDF content (Tampering, B2)

**Threat.** A PDF chunk contains literal `{variable}` sequences (intentional
or accidental). When LangChain historically concatenated chunks into a prompt
template and called `str.format()`, this raised `KeyError` at best and leaked
other template variables at worst.

**Mitigation.** All retrieved chunk content is passed through `escape_braces`
before being inserted into any prompt context.

- Defined: `src/rag/sanitize.py::escape_braces`
- Applied: `src/rag/generation.py::format_documents`
- Tested: `tests/test_sanitize.py::test_escape_braces_prevents_template_injection_via_pdf_content`,
  `test_escape_braces_survives_raw_str_format_when_embedded_in_template`

### T2 — Oversized input → cost / DoS (Denial-of-service, B1+B2+B3)

**Threat.** A user pastes a 10MB question, or a retrieval misconfiguration
returns 500 chunks, and the full blob is sent to a billed LLM.

**Mitigation.** Hard caps at every boundary:

- `MAX_QUESTION_CHARS = 4000` — enforced on CLI entry and on both chain
  entrypoints (`generation.py::chain_fn`, `generation.py::self_rag_fn`).
- `MAX_CONTEXT_CHARS = 120_000` — enforced by `format_documents` as a
  backstop against retriever misconfiguration.
- `MAX_CHUNK_CHARS = 20_000` — documented cap for a single chunk
  (enforced at ingest).

Defined: `src/rag/sanitize.py::clamp_text` + constants.

### T3 — Secret leakage in logs/traces (Information disclosure, B5)

**Threat.** A careless `log.info(f"calling with key={settings.llm_api_key}")`
emits a credential into CI logs or a log aggregator.

**Mitigation.** Every log record passes through `_RedactFilter`, which runs
`redact_secrets` over the message, args, and exception text.

- Defined: `src/rag/sanitize.py::redact_secrets`,
  `src/rag/logging_setup.py::_RedactFilter`
- Installed globally on first logger construction
  (`logging_setup.py::configure_logging`)
- Tested: `tests/test_sanitize.py::test_redact_*`

Additionally, gitleaks (pre-commit + CI) scans for keys in committed files.

### T4 — Malicious pickle on disk (Elevation-of-privilege, B4)

**Threat.** An attacker with write access to the data directory swaps
`bm25.pkl` for a pickle payload that executes arbitrary code when loaded.

**Mitigation.**

- Magic header `b"RAG-BM25v1\n"` written by ingestion; retrieval refuses to
  load files without it.
  - `src/rag/ingest.py` (write path) and `src/rag/retrieval.py::load_bm25_index`
    (read path).
  - Documented in `load_bm25_index` docstring: the header is not a defense
    against an attacker with write access, but prevents accidental loading
    of arbitrary `.pkl` files.
- Ruff `S301` / Bandit `B301` suppressions are explicit (`# noqa: S301 # nosec B301`)
  with a comment justifying the gate.
- Running user is advised to regenerate the index rather than inheriting
  one from an untrusted source.

### T5 — Malformed LLM JSON crashes the grader (Denial-of-service, B3)

**Threat.** An LLM returns ```json\n...\n``` fences, prose before JSON, or
invalid JSON entirely. A naive `json.loads` raises into the user's query
path.

**Mitigation.** `safe_json_loads` strips markdown fences and leading prose,
and returns a caller-supplied fallback on any parse error.

- Defined: `src/rag/sanitize.py::safe_json_loads`
- Applied: `src/rag/generation.py::_grade_documents`, `_check_hallucination`
- Tested: `tests/test_sanitize.py::test_safe_json_loads_*`

### T6 — Dependency CVEs (Tampering, supply chain)

**Threat.** A transitive dependency ships a known CVE that affects our code
path (e.g., a YAML loader, a PDF parser, a serialization library).

**Mitigation.**

- `pip-audit --strict` runs in CI on every push/PR
  (`.github/workflows/ci.yml::security`).
- `bandit -r src/` runs static analysis for common Python security
  anti-patterns.
- Dependencies are pinned in `pyproject.toml` with a floor only; the
  lockfile is regenerated deliberately (not on every CI run) so that
  pip-audit findings point to an actionable version bump.

### T7 — Credentials in commit history (Information disclosure)

**Threat.** A contributor pastes an `.env` file or hard-codes a key.

**Mitigation.**

- `.gitleaks.toml` + `pre-commit` hook blocks commits containing key-shaped
  strings locally.
- Same scan runs in CI (`gitleaks-action`) on full history so a forced push
  can't smuggle past the hook.

### T8 — Unverified LLM answer accepted (Tampering, B3)

**Threat.** The LLM hallucinates a claim that is not supported by retrieved
context, and the user treats it as authoritative.

**Mitigation.** Self-RAG reflection loop when enabled in `config.yaml`:

- Document grading removes irrelevant chunks.
- Hallucination check compares answer against context; if `grounded == "no"`,
  regenerates with a stricter system prompt.
- Relevance check appends a user-visible note when the answer does not
  address the question.

Code: `src/rag/generation.py::_build_self_rag_chain`. This is a *mitigation*,
not a proof — users should still cross-check claims.

## 5. Accepted risks

These are known limitations that we are not currently mitigating:

- **Malicious LLM output**: we treat the LLM as an untrusted external service
  (B3), but we do render its output back to the user. A sufficiently clever
  provider could render control sequences in the terminal. Rich handles
  styled output defensively; raw ANSI injection from the LLM is possible in
  principle and unaddressed.
- **ChromaDB tampering**: we do not sign or hash the vector store. An
  attacker with write access to `settings.chroma_path` can substitute
  embeddings and influence retrieval. This is treated as out-of-scope
  (physical/host compromise).
- **Eval harness authenticity**: RAGAS metrics are computed client-side with
  LLM-as-judge; they are subject to the same hallucination risks as the
  generation step, which is why evaluation runs use a separate, fixed model
  (`gpt-4o-mini`, see commit `2f39b33`).
- **Rate-limiting / quota**: the CLI does not implement client-side
  rate-limiting. An over-enthusiastic Self-RAG loop (retries × grading calls
  × hallucination checks) can rack up billable calls. `self_rag_max_retries`
  in `config.yaml` is the only lever.

## 6. How to update this document

- Each new feature that touches a trust boundary should either map to an
  existing threat row or add a new one.
- When a mitigation is moved or renamed, update the file/function
  references in §4 — the reviewer contract is that every `::name` pointer
  still resolves.
- Record new accepted risks in §5 rather than silently.
