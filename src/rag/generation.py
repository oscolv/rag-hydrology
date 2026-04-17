"""RAG generation chain with citation support."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

from rag.config import Settings

SYSTEM_PROMPT = """You are a research assistant specializing in water resources, \
GRACE satellite data, and hydrology. Answer questions using ONLY the provided context.

Rules:
- Cite sources using [Source: filename, Page: N] format after each claim
- If the context doesn't contain the answer, say so explicitly
- For Spanish-language sources, translate relevant content when answering in English
- Preserve technical accuracy, especially for equations, units, and measurements
- When discussing GRACE data, be precise about spatial/temporal resolution
- Structure longer answers with clear paragraphs

Context:
{context}"""


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


def format_documents(docs: list[Document]) -> str:
    """Format retrieved documents with source citations for the prompt."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever: BaseRetriever, settings: Settings):
    """Build a RAG chain that returns only the answer string."""
    llm = _build_llm(settings)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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
    llm = _build_llm(settings)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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
