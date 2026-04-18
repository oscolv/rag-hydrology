"""RAG generation chain with citation support and Self-RAG reflection.

Self-RAG (Self-Reflective RAG) adds three verification steps:
1. Document Grading — filter irrelevant retrieved documents
2. Hallucination Check — verify answer is grounded in context
3. Answer Relevance — verify answer addresses the question
If checks fail, the system reformulates and retries.
"""

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from rich.console import Console

from rag.config import Settings
from rag.factories import get_chat_llm
from rag.logging_setup import get_logger, new_request_id
from rag.sanitize import (
    MAX_CONTEXT_CHARS,
    MAX_QUESTION_CHARS,
    clamp_text,
    escape_braces,
    safe_json_loads,
)
from rag.tracing import trace_config


def _trace_meta(settings: Settings, rid: str, **extra) -> dict:
    """Build the metadata dict attached to every Langfuse trace for a query."""
    meta = {
        "request_id": rid,
        "collection": settings.active_collection,
        "model": settings.llm.model,
    }
    meta.update(extra)
    return meta

log = get_logger(__name__)

console = Console()

# System prompt is resolved from settings.domain.system_prompt at runtime
# (see _system_prompt below). The template must contain "{context}".


# ---------------------------------------------------------------------------
# LLM Builder
# ---------------------------------------------------------------------------


def _system_prompt(settings: Settings) -> str:
    """Return the configured system prompt, validating {context} placeholder."""
    tpl = settings.domain.system_prompt
    if "{context}" not in tpl:
        raise ValueError(
            "domain.system_prompt must contain the '{context}' placeholder"
        )
    return tpl


def _build_llm(settings: Settings) -> ChatOpenAI:
    """Return the shared LLM client (cached across calls)."""
    return get_chat_llm(settings)


# ---------------------------------------------------------------------------
# Document Formatting
# ---------------------------------------------------------------------------


def extract_citation_numbers(answer: str) -> list[int]:
    """Return the set of [N] citations referenced in the answer, in order of first use."""
    import re
    seen: dict[int, None] = {}
    for match in re.finditer(r"\[(\d{1,3})\]", answer):
        n = int(match.group(1))
        if n not in seen:
            seen[n] = None
    return list(seen.keys())


def format_documents(docs: list[Document]) -> str:
    """Format retrieved documents with numbered + named citations for the prompt.

    Each document is prefixed with `[N]` (rank) AND the original
    `[Source: filename, Page: N]` tag, so the LLM can cite with either
    `[1][2]` (numeric, preferred for inline display) or
    `[Source: filename, Page: N]` (legacy, still supported).

    Chunk content is passed through escape_braces so literal `{...}` sequences
    inside a PDF can't collide with ChatPromptTemplate's str.format() pass.
    The concatenated context is also clamped to MAX_CONTEXT_CHARS as a
    backstop against retriever misconfiguration.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        safe_content = escape_braces(doc.page_content)
        formatted.append(f"[{i}] [Source: {source}, Page: {page}]\n{safe_content}")
    return clamp_text("\n\n---\n\n".join(formatted), MAX_CONTEXT_CHARS)


# ---------------------------------------------------------------------------
# Self-RAG: Document Grading
# ---------------------------------------------------------------------------

_GRADE_PROMPT = """You are a relevance grader for {domain_description}.

Given a user question and retrieved documents, determine which documents contain \
information relevant to answering the question.

Question: {question}

Documents:
{documents}

For each document (numbered), respond with "yes" if it is relevant or "no" if it is not.
Respond ONLY as a JSON array of "yes"/"no" strings matching the document order.
Example: ["yes", "no", "yes"]"""

_REFORMULATE_PROMPT = """The retrieved documents were not relevant to the question. \
Reformulate the question to improve retrieval. Keep the same intent but use different \
terms, synonyms, or a more specific/general phrasing.

Original question: {question}

Respond ONLY with the reformulated question, nothing else."""

_HALLUCINATION_PROMPT = """You are a grading assistant for a RAG system. Evaluate the \
generated answer against the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Evaluate:
1. "grounded": Is every claim in the answer supported by the context? ("yes" or "no")
2. "relevant": Does the answer actually address the question? ("yes" or "no")
3. "issues": Brief description of any problems found (empty string if none)

Respond ONLY as JSON: {{"grounded": "yes"|"no", "relevant": "yes"|"no", "issues": ""}}"""


def _grade_documents(
    llm: ChatOpenAI,
    question: str,
    docs: list[Document],
    domain_description: str = "a research RAG system",
    config: dict | None = None,
) -> list[Document]:
    """Grade each document's relevance to the question. Return only relevant docs."""
    if not docs:
        return []

    # Build numbered document list for the prompt
    doc_texts = []
    for i, doc in enumerate(docs, 1):
        excerpt = doc.page_content[:500]
        source = doc.metadata.get("source", "unknown")
        doc_texts.append(f"[{i}] (Source: {source})\n{excerpt}")

    prompt = ChatPromptTemplate.from_messages([("human", _GRADE_PROMPT)])
    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "question": question,
            "documents": "\n\n".join(doc_texts),
            "domain_description": domain_description,
        }, config=config or {})
    except Exception:
        return docs  # On LLM error, keep all documents

    grades = safe_json_loads(result, fallback=[])
    if not isinstance(grades, list):
        return docs
    relevant = [
        doc for doc, grade in zip(docs, grades, strict=False)
        if str(grade).lower().strip() == "yes"
    ]
    return relevant if relevant else docs[:2]  # Fallback: keep top 2


def _reformulate_query(llm: ChatOpenAI, question: str, config: dict | None = None) -> str:
    """Reformulate a query when retrieved documents are not relevant."""
    prompt = ChatPromptTemplate.from_messages([("human", _REFORMULATE_PROMPT)])
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"question": question}, config=config or {}).strip()
    except Exception:
        return question


def _check_hallucination(
    llm: ChatOpenAI,
    question: str,
    context: str,
    answer: str,
    config: dict | None = None,
) -> dict:
    """Check if the answer is grounded and relevant. Returns grading dict."""
    prompt = ChatPromptTemplate.from_messages([("human", _HALLUCINATION_PROMPT)])
    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        }, config=config or {})
    except Exception:
        return {"grounded": "yes", "relevant": "yes", "issues": ""}

    parsed = safe_json_loads(
        result, fallback={"grounded": "yes", "relevant": "yes", "issues": ""},
    )
    if not isinstance(parsed, dict):
        return {"grounded": "yes", "relevant": "yes", "issues": ""}
    return parsed


# ---------------------------------------------------------------------------
# Standard RAG Chains
# ---------------------------------------------------------------------------


def build_rag_chain(retriever: BaseRetriever, settings: Settings):
    """Build a RAG chain that returns only the answer string."""
    if settings.retrieval.self_rag:
        return _build_self_rag_chain(retriever, settings, with_sources=False)

    llm = _build_llm(settings)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _system_prompt(settings)),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def build_rag_chain_with_sources(retriever: BaseRetriever, settings: Settings):
    """Build a RAG chain that returns both answer and source documents.

    Returns dict with keys: "answer" (str), "source_documents" (list[Document])
    """
    if settings.retrieval.self_rag:
        return _build_self_rag_chain(retriever, settings, with_sources=True)

    llm = _build_llm(settings)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _system_prompt(settings)),
        ("human", "{question}"),
    ])

    def chain_fn(question: str) -> dict:
        rid = new_request_id()
        question = clamp_text(question, MAX_QUESTION_CHARS)
        cfg = trace_config("rag.query", _trace_meta(settings, rid))
        log.info("rag.query.start", extra={"rid": rid, "q_len": len(question)})
        docs = retriever.invoke(question, config=cfg)
        context = format_documents(docs)
        log.info("rag.retrieved", extra={"rid": rid, "doc_count": len(docs)})
        messages = prompt.invoke({
            "context": context,
            "question": question,
        })
        answer = llm.invoke(messages, config=cfg)
        log.info(
            "rag.answered",
            extra={"rid": rid, "answer_len": len(answer.content)},
        )
        return {
            "answer": answer.content,
            "source_documents": docs,
            "question": question,
            "request_id": rid,
        }

    return chain_fn


# ---------------------------------------------------------------------------
# Self-RAG Chain
# ---------------------------------------------------------------------------


def _build_self_rag_chain(
    retriever: BaseRetriever,
    settings: Settings,
    with_sources: bool = False,
):
    """Build a Self-RAG chain with document grading, hallucination check, and retry.

    Flow:
    1. Retrieve documents
    2. Grade documents for relevance → filter irrelevant
    3. If too few relevant docs → reformulate query → retry (up to max_retries)
    4. Generate answer
    5. Check hallucination + relevance
    6. If hallucination detected → regenerate with stricter prompt
    7. Return answer (optionally with sources and reflection metadata)
    """
    llm = _build_llm(settings)
    # Use a fast/cheap model for grading if available, else same model
    grader_llm = _build_llm(settings)
    max_retries = settings.retrieval.self_rag_max_retries
    system_prompt_text = _system_prompt(settings)
    grader_description = settings.domain.grader_description

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "{question}"),
    ])

    stricter_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text + "\n\nIMPORTANT: A previous answer was found to "
         "contain unsupported claims. Be extra careful to ONLY state facts that are "
         "explicitly present in the context. If unsure, say you don't have enough "
         "information."),
        ("human", "{question}"),
    ])

    def self_rag_fn(question: str) -> dict | str:
        rid = new_request_id()
        question = clamp_text(question, MAX_QUESTION_CHARS)
        cfg = trace_config("rag.query.self_rag", _trace_meta(settings, rid))
        reflection_log = []
        current_query = question
        relevant_docs = []

        # --- Phase 1: Retrieve & Grade (with retry) ---
        for attempt in range(max_retries + 1):
            docs = retriever.invoke(current_query, config=cfg)
            relevant_docs = _grade_documents(
                grader_llm, question, docs,
                domain_description=grader_description, config=cfg,
            )

            relevance_ratio = len(relevant_docs) / max(len(docs), 1)
            reflection_log.append({
                "step": "grade_documents",
                "attempt": attempt + 1,
                "query": current_query,
                "retrieved": len(docs),
                "relevant": len(relevant_docs),
                "ratio": round(relevance_ratio, 2),
            })

            # If we have enough relevant docs, proceed
            if relevance_ratio >= 0.3 or len(relevant_docs) >= 2:
                break

            # Otherwise, reformulate and retry
            if attempt < max_retries:
                current_query = _reformulate_query(grader_llm, current_query, config=cfg)
                reflection_log.append({
                    "step": "reformulate",
                    "original": question,
                    "reformulated": current_query,
                })

        # --- Phase 2: Generate Answer ---
        context = format_documents(relevant_docs)
        messages = prompt.invoke({
            "context": context,
            "question": question,  # Always use original question for generation
        })
        answer = llm.invoke(messages, config=cfg).content

        reflection_log.append({
            "step": "generate",
            "context_docs": len(relevant_docs),
        })

        # --- Phase 3: Hallucination & Relevance Check ---
        grading = _check_hallucination(grader_llm, question, context, answer, config=cfg)
        reflection_log.append({
            "step": "hallucination_check",
            **grading,
        })

        # If hallucination detected, regenerate with stricter prompt
        if grading.get("grounded") == "no":
            messages = stricter_prompt.invoke({
                "context": context,
                "question": question,
            })
            answer = llm.invoke(messages, config=cfg).content
            reflection_log.append({
                "step": "regenerate",
                "reason": "hallucination detected",
            })

            # Re-check after regeneration
            grading = _check_hallucination(
                grader_llm, question, context, answer, config=cfg,
            )
            reflection_log.append({
                "step": "hallucination_recheck",
                **grading,
            })

        # If answer is not relevant, append a note
        if grading.get("relevant") == "no":
            answer += (
                "\n\n*Note: The retrieved context may not fully address this question. "
                "Consider rephrasing or checking if the relevant documents are indexed.*"
            )

        if with_sources:
            return {
                "answer": answer,
                "source_documents": relevant_docs,
                "question": question,
                "request_id": rid,
                "reflection": reflection_log,
            }
        else:
            return answer

    return self_rag_fn


# ---------------------------------------------------------------------------
# Streaming chains (yield events for SSE / CLI live display)
# ---------------------------------------------------------------------------
#
# Event schema:
#   {"event": "retrieval_start"}                    — pipeline started
#   {"event": "sources", "documents": [Document]}   — retrieved + filtered docs
#   {"event": "reflection", "step": {...}}          — Self-RAG log entries
#   {"event": "token", "content": "..."}            — one token / chunk
#   {"event": "done", "answer": "...", ...}         — final answer + metadata
#   {"event": "error", "message": "..."}            — failure
#
# This shape is stable: it's consumed by the CLI (Live panel) and the web UI
# (Server-Sent Events). Don't repurpose event names without bumping the web
# client.


def _stream_llm_response(
    llm: ChatOpenAI, messages, config: dict | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield (full_text_so_far, new_chunk) pairs from the LLM stream."""
    buffer = ""
    for chunk in llm.stream(messages, config=config or {}):
        piece = chunk.content or ""
        if not piece:
            continue
        buffer += piece
        yield buffer, piece


def build_rag_chain_streaming(retriever: BaseRetriever, settings: Settings):
    """Streaming RAG: emits retrieval + token-by-token answer events.

    Returns a function `fn(question) -> Iterator[dict]` that yields typed
    events. Self-RAG is supported: grading/reformulation run before the stream
    begins, and the final generation phase is what gets streamed.
    """
    if settings.retrieval.self_rag:
        return _build_self_rag_streaming(retriever, settings)

    llm = _build_llm(settings)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _system_prompt(settings)),
        ("human", "{question}"),
    ])

    def stream_fn(question: str) -> Iterator[dict]:
        rid = new_request_id()
        question = clamp_text(question, MAX_QUESTION_CHARS)
        cfg = trace_config("rag.query.stream", _trace_meta(settings, rid))
        log.info("rag.stream.start", extra={"rid": rid, "q_len": len(question)})

        yield {"event": "retrieval_start", "request_id": rid}

        try:
            docs = retriever.invoke(question, config=cfg)
        except Exception as e:
            log.warning("rag.retrieval.failed", extra={"rid": rid, "err": str(e)})
            yield {"event": "error", "message": f"Retrieval failed: {e}"}
            return

        yield {"event": "sources", "documents": docs, "request_id": rid}

        context = format_documents(docs)
        messages = prompt.invoke({"context": context, "question": question})

        answer = ""
        try:
            for full, piece in _stream_llm_response(llm, messages, config=cfg):
                answer = full
                yield {"event": "token", "content": piece}
        except Exception as e:
            log.warning("rag.stream.failed", extra={"rid": rid, "err": str(e)})
            yield {"event": "error", "message": f"Generation failed: {e}"}
            return

        log.info(
            "rag.stream.done",
            extra={"rid": rid, "answer_len": len(answer), "docs": len(docs)},
        )
        yield {
            "event": "done",
            "answer": answer,
            "source_documents": docs,
            "question": question,
            "request_id": rid,
        }

    return stream_fn


def _build_self_rag_streaming(retriever: BaseRetriever, settings: Settings):
    """Self-RAG flavor of the streaming chain.

    The retrieve-grade-reformulate loop runs eagerly (not streamable), emitting
    reflection events as it goes. Only the final generation is streamed. If
    hallucination is detected, the regenerated answer is streamed as a fresh
    series of token events after a reflection event explaining the retry.
    """
    llm = _build_llm(settings)
    grader_llm = _build_llm(settings)
    max_retries = settings.retrieval.self_rag_max_retries
    system_prompt_text = _system_prompt(settings)
    grader_description = settings.domain.grader_description

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "{question}"),
    ])
    stricter_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text + "\n\nIMPORTANT: A previous answer was found to "
         "contain unsupported claims. Be extra careful to ONLY state facts that are "
         "explicitly present in the context. If unsure, say you don't have enough "
         "information."),
        ("human", "{question}"),
    ])

    def stream_fn(question: str) -> Iterator[dict]:
        rid = new_request_id()
        question = clamp_text(question, MAX_QUESTION_CHARS)
        cfg = trace_config("rag.query.self_rag.stream", _trace_meta(settings, rid))
        yield {"event": "retrieval_start", "request_id": rid}

        reflection_log: list[dict] = []
        current_query = question
        relevant_docs: list[Document] = []

        for attempt in range(max_retries + 1):
            try:
                docs = retriever.invoke(current_query, config=cfg)
            except Exception as e:
                yield {"event": "error", "message": f"Retrieval failed: {e}"}
                return

            relevant_docs = _grade_documents(
                grader_llm, question, docs,
                domain_description=grader_description, config=cfg,
            )
            ratio = len(relevant_docs) / max(len(docs), 1)
            step = {
                "step": "grade_documents",
                "attempt": attempt + 1,
                "query": current_query,
                "retrieved": len(docs),
                "relevant": len(relevant_docs),
                "ratio": round(ratio, 2),
            }
            reflection_log.append(step)
            yield {"event": "reflection", "step": step}

            if ratio >= 0.3 or len(relevant_docs) >= 2:
                break

            if attempt < max_retries:
                current_query = _reformulate_query(grader_llm, current_query, config=cfg)
                step = {
                    "step": "reformulate",
                    "original": question,
                    "reformulated": current_query,
                }
                reflection_log.append(step)
                yield {"event": "reflection", "step": step}

        yield {"event": "sources", "documents": relevant_docs, "request_id": rid}

        context = format_documents(relevant_docs)
        messages = prompt.invoke({"context": context, "question": question})

        answer = ""
        try:
            for full, piece in _stream_llm_response(llm, messages, config=cfg):
                answer = full
                yield {"event": "token", "content": piece}
        except Exception as e:
            yield {"event": "error", "message": f"Generation failed: {e}"}
            return

        reflection_log.append({"step": "generate", "context_docs": len(relevant_docs)})

        # Hallucination check (post-stream, synchronous)
        grading = _check_hallucination(grader_llm, question, context, answer, config=cfg)
        hallu_step = {"step": "hallucination_check", **grading}
        reflection_log.append(hallu_step)
        yield {"event": "reflection", "step": hallu_step}

        if grading.get("grounded") == "no":
            retry_step = {"step": "regenerate", "reason": "hallucination detected"}
            reflection_log.append(retry_step)
            yield {"event": "reflection", "step": retry_step}
            yield {"event": "regenerating"}

            messages = stricter_prompt.invoke({"context": context, "question": question})
            answer = ""
            try:
                for full, piece in _stream_llm_response(llm, messages, config=cfg):
                    answer = full
                    yield {"event": "token", "content": piece, "regenerated": True}
            except Exception as e:
                yield {"event": "error", "message": f"Regeneration failed: {e}"}
                return

            grading = _check_hallucination(
                grader_llm, question, context, answer, config=cfg,
            )
            recheck_step = {"step": "hallucination_recheck", **grading}
            reflection_log.append(recheck_step)
            yield {"event": "reflection", "step": recheck_step}

        if grading.get("relevant") == "no":
            note = (
                "\n\n*Note: The retrieved context may not fully address this question. "
                "Consider rephrasing or checking if the relevant documents are indexed.*"
            )
            answer += note
            yield {"event": "token", "content": note}

        yield {
            "event": "done",
            "answer": answer,
            "source_documents": relevant_docs,
            "question": question,
            "request_id": rid,
            "reflection": reflection_log,
        }

    return stream_fn
