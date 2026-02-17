# assistant/qa.py

from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are an assistant for graduate students in the Advanced Data Analytics program at UNT.

Use ONLY the information in the context to answer the question.
If the answer is not clearly stated in the context, say:
"I couldn't find a clear answer in the provided documents."

When possible:
- mention whether the rule comes from the ADTA Graduate Handbook or the Student Academic Integrity Policy.
- include short citations like (source: filename, page X).

Question:
{question}

Context:
{context}
"""
)


def format_context(docs: List[Document]) -> str:
    """Format retrieved docs as a readable context block with source & page."""
    parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "N/A")
        parts.append(f"[{source}, page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    vector_store: InMemoryVectorStore,
    question: str,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve top-k relevant chunks and ask the LLM to answer
    based only on those chunks.
    """
    # 1. Retrieval: convert question -> embedding and find similar chunks
    retrieved_docs = vector_store.similarity_search(question, k=k)

    # 2. Build context string
    context_str = format_context(retrieved_docs)

    # 3. Call LLM with question + context
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
    messages = ANSWER_PROMPT.format_messages(
        question=question,
        context=context_str,
    )
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": retrieved_docs,
    }
