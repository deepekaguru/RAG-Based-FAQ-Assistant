# main.py

from assistant.ingest import ingest_corpus
from assistant.qa import answer_question


def main():
    # 1. Build vector store from your two PDFs
    vector_store = ingest_corpus()

    print("\n🎓 ADTA RAG Assistant")
    print("Ask about degree requirements, policies, or academic integrity.")
    print("Type 'exit' to quit.\n")

    # 2. Simple chat loop
    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        result = answer_question(vector_store, question)
        print("\nANSWER:\n")
        print(result["answer"])

        print("\nSOURCES USED:\n")
        for doc in result["sources"]:
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "N/A")
            print(f"- {src} (page {page})")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
