# demo_teaching_agent.py

def main():
    print("🚀 Starting Demo Teaching Agent...")

    # Simulate LLM response
    response = "Hello! This is a demo response from LLM."
    print("LLM Response:", response)

    # Simulate embeddings
    sample_text = "This is a test sentence."
    embedding = [0.1] * 10  # fake embedding
    print("Embedding length:", len(embedding))

    # Simulate vector store
    vector_store = {"doc1": sample_text}
    query = "test sentence"
    results = [k for k, v in vector_store.items() if query in v]
    print("Vector store search results:", results)

if __name__ == "__main__":
    main()
