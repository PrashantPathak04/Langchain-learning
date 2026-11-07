# LangChain Learning

A Python learning project for building a helper app that lets users upload a
document and ask questions about its contents.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/) for running a local language model

## Set up the environment

From the project directory, run:

```powershell
uv sync
```

This creates `.venv`, installs the project dependencies, and generates or
updates `uv.lock`. To activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You can also run project commands without activating it by prefixing them with
`uv run`.

## Run the app

Install [Ollama](https://ollama.com/), then download the default chat and
embedding models:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Start Ollama, then launch the app:

```powershell
uv run streamlit run app.py
```

Open the local URL printed by Streamlit, upload a PDF, DOCX, TXT, or Markdown
file, and select **Index document**. Once indexing completes, ask questions in
the chat box. The app uses Ollama locally for embeddings and answers.
