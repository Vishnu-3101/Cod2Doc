# 🧠 Code to Documentation Generator (LLM-Powered)

Turn any Python repository into beautifully structured, developer-friendly documentation — automatically.
This tool analyzes your codebase, builds a dependency graph, retrieves related components, and uses LLMs to generate top-down, human-readable documentation.

Now with an **interactive Streamlit UI**, **GitHub repo cloning**, and **step-by-step animated generation**.

---

## 🚀 Features

✅ **Automatic Repository Analysis**

* Paste a GitHub repo link — the app clones it locally and parses all Python files.

✅ **Dependency Graph Construction**

* Uses AST parsing to build a detailed dependency graph between functions, classes, and files.

✅ **Multiple Entry Points**

* Detects all code entry points (e.g., main scripts) and generates **separate docs** for each.

✅ **Step-by-Step LLM Documentation**

* Feeds code to the LLM in a top-down order with short-term memory, ensuring context awareness without hitting token limits.

✅ **Interactive Streamlit UI**

* Real-time progress animation:
  *“Cloning repo → Building graph → Finding entry points → Generating docs → Compiling results”*

✅ **Downloadable Docs**

* Each generated document can be previewed in the app and downloaded in Markdown format.

---

## 🧩 Project Structure

```text

├── main.py                     # streamlit pipeline
├── utils/
│   ├── build_graph.py          # Builds dependency graph via AST parsing
│   ├── loader.py               # Loads docs, retrieves code with dependencies
│   ├── parser.py               # Extracts functions/classes from code
│   ├── toposort.py             # Handles graph traversal and sorting
├── docgen/                     # Core documentation generation pipeline
│   ├── entrypoints.py          # Identifies and manages entry points in the dependency graph
│   ├── generator.py            # Coordinates the doc generation process for each entry point
│   ├── retriever.py            # Retrieves dependent code snippets and context for doc generation
├── llm/                        # LLM integration and chain setup
│   └── chain_setup.py          # Defines and initializes LLM chains, memory, and retrievers
├── prompts/                    # Organized prompt templates for LLM interactions
│   └── doc_prompts.py          # Contains detailed and structured prompts for documentation generation
├── output/
│   ├── dependency_graph.json   # Auto-generated dependency graph
│   └── documentation_*.md      # Generated documentation files
```

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/vishnu-3101/cod2doc.git
cd cod2doc
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows (Powershell)
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies include:**

* `langchain`
* `langchain-google-genai`
* `langchain-core`
* `langchain-community`
* `gitpython`
* `streamlit`
* `shutil`
* `pathlib`

---

## 🦙 Setup Gemini API Keys

This project uses **Google Gemini 2.0 Flash** for LLM-powered documentation generation.

### 1. Get Your API Key

* Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
* Sign in with your Google account
* Click **“Create API key”** and copy it.

### 2. Add the Key to Your Environment

You can store the key in a `.env` file (recommended) or set it as an environment variable.

Create a file named `.env` in the project root and add:

```bash
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Running the Streamlit App

Launch the app:

```bash
streamlit run app.py
```

### 💡 What Happens Behind the Scenes

1. You paste a GitHub repository link.
2. The app:

   * Clones the repository locally.
   * Builds a dependency graph.
   * Identifies all entry points.
   * Generates detailed documentation for each entry point.
3. You can **preview**, **download**, or **regenerate** documentation interactively.

---

## 🧠 Example Workflow

1. Enter your repository link in the Streamlit interface:

   ```
   https://github.com/your-username/sample-python-project
   ```
2. The app analyzes the repo:

   * 🔍 “Understanding repo files…”
   * 🧩 “Building dependency graph…”
   * 🚀 “Generating docs…”
3. View the generated documentation for each entry point in Markdown format in output folder:

   ```text
   output
   ├── documentation_{entry_point_id}.md
   ```

---


## 🧭 Roadmap

* [ ] Add multi-model support (OpenAI, Anthropic, Ollama)
* [ ] Add theme customization for output docs
* [ ] Enable multi-language code analysis (JS, Go, C++)
* [ ] Generate architecture diagrams from dependency graphs

---

## 🤝 Contributing

We welcome contributions!
To contribute:

1. Fork this repo
2. Create a new branch
3. Make your changes
4. Submit a PR 🚀

---

## 📜 License

This project is licensed under the **MIT License** — free for personal and commercial use.

