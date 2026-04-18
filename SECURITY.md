# Security Policy

## Reporting a Vulnerability

If you believe you have found a security issue in this RAG pipeline, please **do not open a public GitHub issue**. Instead:

- Email the maintainer at the address listed in `pyproject.toml` / commit history.
- Include a clear description, reproduction steps, and the commit SHA you tested against.
- Expect an initial acknowledgment within **7 days**.

We treat the following as in-scope:

- Prompt-template injection or context-window smuggling through ingested documents
- Secret/credential leakage in logs, traces, error messages, or the BM25 index
- Deserialization issues (the BM25 pickle format, ChromaDB persistence)
- Dependency CVEs flagged by `pip-audit` that are reachable from our code paths
- Authentication/authorization bypasses in the CLI or evaluation harness

Out of scope:

- Denial-of-service via oversized inputs that stay within configured limits
  (`MAX_QUESTION_CHARS`, `MAX_CHUNK_CHARS`, `MAX_CONTEXT_CHARS`)
- Model behavior (hallucinations, biased answers) — use the evaluation harness
  and configuration, not a vulnerability report
- Findings that require an attacker to already have write access to `$HOME`,
  the ChromaDB directory, or `bm25.pkl`

## Supported Versions

Main branch only. Point releases are not maintained. If you are running a
fork or a tagged version, please upgrade to the latest `main` before
reporting.

## Threat Model

The threat model — trust boundaries, assets, and mitigations — is documented
in [`THREAT_MODEL.md`](./THREAT_MODEL.md). Please read it before filing a
report so we can focus the conversation on gaps rather than accepted risks.
