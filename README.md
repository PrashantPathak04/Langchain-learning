# RAG over `vectordb.txt`

The ingestion script splits `vectordb.txt` into chunks and stores them in
Pinecone using its Documents API and integrated embedding. Ask questions using
Pinecone retrieval and an Ollama chat model.

## Setup

1. Install and start [Ollama](https://ollama.com/), then download the default
   chat model:

   ```powershell
   ollama pull qwen3:1.7b
   ```

2. Use a Pinecone document index with an embedding-enabled string field named
   `chunk_text` (the current index schema). Pinecone handles embeddings for
   indexing and search.
3. Set these values in `.env`:

   ```dotenv
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=your-index-name
   ```

   Optional settings are `PINECONE_TEXT_FIELD` (defaults to `chunk_text`),
   `PINECONE_NAMESPACE` (defaults to `__default__`), `OLLAMA_CHAT_MODEL`, and
   `OLLAMA_BASE_URL`.

## Use

Index or refresh the text file after changing it:

```powershell
uv run python ingestion.py index
```

Ask a question against the indexed chunks:

```powershell
uv run python ingestion.py ask "What does this document say about ...?"
```
