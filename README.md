# Code to Documentation Generator

This project analyzes Python code, builds a dependency graph between functions and components, retrieves relevant code snippets, and generates human-readable documentation using LLMs.

It combines **graph-based analysis** with **hybrid retrieval (BM25 + embeddings)** to fetch contextually relevant code and uses **Ollama-powered LLMs** to produce structured documentation.

---

## 🚀 Features

- Build a dependency graph of functions and components in a repo  
- Perform topological sorting & cycle detection for dependency resolution  
- Retrieve code context using a hybrid retriever (BM25 + dense embeddings)  
- Automatically generate documentation for queried code using an LLM  

---

## 📂 Project Structure

```text
.
├── main.py                     
├── utils/
│   ├── build_graph.py          
│   ├── loader.py               # Loads docs, retrieves code with dependencies
│   ├── parser.py               # Extracts functions/classes from source code
│   ├── toposort.py             # Graph algorithms (DFS, Tarjan, topological sort)
├── knowledge_base/             # Source repo/codebase to analyze
├── output/
│   └── dependency_graph.json   # Auto-generated dependency graph
```

## ⚙️ Installation

### Clone the repo
```bash
git clone https://github.com/your-username/code-doc-gen.git
cd code-doc-gen
```

### Create a virtual environment
```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows (Powershell):
venv\Scripts\Activate.ps1
```

### Install dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies include:**
- `langchain`  
- `langchain-community`  
- `langchain-huggingface`  
- `langchain-ollama`  
- `faiss-cpu`  
- `rank-bm25`  
- `transformers`  

---

## Install and run Ollama

- Follow Ollama setup instructions: [Ollama.ai](https://ollama.ai)  
- Pull the required model (example uses `qwen3:0.6b`):
```bash
ollama pull qwen3:0.6b
```


## ▶️ Running the Project

1. Place your target source code inside the `knowledge_base/` folder.  
2. Run the main script:
   ```bash
   python main.py
3. On first run, it generates output/dependency_graph.json.

4. The retrievers fetch code relevant to the query.

5. The LLM outputs generated documentation in structured paragraphs.


## 🔍 Example Usage

Inside `main.py`, set the query:
```python
query_code = "def backward():"
```

#### When executed, the pipeline:

* Finds the backward() function in the repo
* Expands its context by retrieving dependencies
* Sends it to the LLM (qwen3:0.6b)
* Produces documentation

## Example Output
```text
backward() is a function responsible for computing gradients in the training loop.
It depends on forward propagation results and updates model parameters accordingly.

In detail, backward() leverages loss computations and calls gradient update functions.
Its dependencies include optimizer utilities and helper functions defined in optimizer.py.
```