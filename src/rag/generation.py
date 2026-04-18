"""RAG generation chain with citation support and Self-RAG reflection.

Self-RAG (Self-Reflective RAG) adds three verification steps:
1. Document Grading — filter irrelevant retrieved documents
2. Hallucination Check — verify answer is grounded in context
3. Answer Relevance — verify answer addresses the question
If checks fail, the system reformulates and retries.
"""

import json

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from rich.console import Console

from rag.config import Settings

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
    """Build the LLM instance, supporting OpenRouter via base_url."""
    kwargs = {
        "model": settings.llm.model,
        "temperature": settings.llm.temperature,
        "openai_api_key": settings.llm_api_key,
    }
    if settings.llm_base_url:
        kwargs["openai_api_base"] = settings.llm_base_url
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Document Formatting
# ---------------------------------------------------------------------------


def format_documents(docs: list[Document]) -> str:
    """Format retrieved documents with source citations for the prompt."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


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
        })
        # Parse JSON array
        grades = json.loads(result.strip())
        relevant = [
            doc for doc, grade in zip(docs, grades)
            if str(grade).lower().strip() == "yes"
        ]
        return relevant if relevant else docs[:2]  # Fallback: keep top 2
    except Exception:
        return docs  # On error, keep all documents


def _reformulate_query(llm: ChatOpenAI, question: str) -> str:
    """Reformulate a query when retrieved documents are not relevant."""
    prompt = ChatPromptTemplate.from_messages([("human", _REFORMULATE_PROMPT)])
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"question": question}).strip()
    except Exception:
        return question


def _check_hallucination(
    llm: ChatOpenAI,
    question: str,
    context: str,
    answer: str,
) -> dict:
    """Check if the answer is grounded and relevant. Returns grading dict."""
    prompt = ChatPromptTemplate.from_messages([("human", _HALLUCINATION_PROMPT)])
    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        })
        return json.loads(result.strip())
    except Exception:
        return {"grounded": "yes", "relevant": "yes", "issues": ""}


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

    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        return {"docs": docs, "context": format_documents(docs)}

    def chain_fn(question: str) -> dict:
        retrieval = retrieve_and_format(question)
        messages = prompt.invoke({
            "context": retrieval["context"],
            "question": question,
        })
        answer = llm.invoke(messages)
        return {
            "answer": answer.content,
            "source_documents": retrieval["docs"],
            "question": question,
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
        reflection_log = []
        current_query = question
        relevant_docs = []

        # --- Phase 1: Retrieve & Grade (with retry) ---
        for attempt in range(max_retries + 1):
            docs = retriever.invoke(current_query)
            relevant_docs = _grade_documents(
                grader_llm, question, docs, domain_description=grader_description,
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
                current_query = _reformulate_query(grader_llm, current_query)
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
        answer = llm.invoke(messages).content

        reflection_log.append({
            "step": "generate",
            "context_docs": len(relevant_docs),
        })

        # --- Phase 3: Hallucination & Relevance Check ---
        grading = _check_hallucination(grader_llm, question, context, answer)
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
            answer = llm.invoke(messages).content
            reflection_log.append({
                "step": "regenerate",
                "reason": "hallucination detected",
            })

            # Re-check after regeneration
            grading = _check_hallucination(grader_llm, question, context, answer)
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
                "reflection": reflection_log,
            }
        else:
            return answer

    return self_rag_fn
