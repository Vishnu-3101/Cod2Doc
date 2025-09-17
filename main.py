import sys
import logging
import os
import json
from utils.build_graph import BuildGraph
from utils.loader import retrieve_with_dependencies, get_doc
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("docstring_generator")

def find_entrypoints(dependency_graph_path):
    # Load the graph
    with open(dependency_graph_path, "r") as f:
        graph = json.load(f)

    # Track dependencies
    depends_on_map = {k: set(v["depends_on"]) for k, v in graph.items()}

    # Build reverse dependencies (who depends on me)
    depended_by_map = {k: set() for k in graph}
    for comp, deps in depends_on_map.items():
        for dep in deps:
            if dep in depended_by_map:
                depended_by_map[dep].add(comp)

    # Find entrypoints:
    # Components that depend on others, but no one depends on them
    entrypoints = []
    for comp, deps in depends_on_map.items():
        if deps and len(depended_by_map[comp]) == 0:
            entrypoints.append(comp)
    
    # Filter components where "main" appears in the ID
    main_components = [comp_id for comp_id in entrypoints if "main" in comp_id.lower()]

    return main_components


dependency_graph_path = "output/dependency_graph.json"
repo_path = "knowledge_base"

if not os.path.exists(dependency_graph_path):
    BuildGraph(repo_path=repo_path, dependency_graph_path=dependency_graph_path)

else:
    print("Dependency graph already created. Using it for further processing....")

entry_points = find_entrypoints(dependency_graph_path)

print(entry_points)


# docs, data = get_doc(dependency_graph_path)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

# vectorstore = FAISS.from_documents(docs, embeddings)
# embedding_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # BM25 retriever
# bm25_retriever = BM25Retriever.from_documents(docs)
# bm25_retriever.k = 2

# # Hybrid retriever (weighted ensemble)
# hybrid_retriever = EnsembleRetriever(
#     retrievers=[embedding_retriever, bm25_retriever],
#     weights=[0.3, 0.7] 
# )

# # Example
# query_code = "def backward():"
# results,expanded_results = retrieve_with_dependencies(query_code, hybrid_retriever, data)


# # for result in results:d
# #     # print(result.metadata['score'])
# #     print(result.page_content)
# #     print("------------------------------------")

# # print(expanded_results)

# for res in expanded_results:
#     print(res["id"], "=> depends on", res["depends_on"])
#     print("------------------------------------")


# llm = ChatOllama(model="qwen3:0.6b")

# template = """
# You are a code documentation generator.
# Generate documentation for the {query_code}. Use {components} for any additional information 
# required to generate detailed documentation. The documentation should be of two paragraphs.:

# 1. Brief introduction involving functionality and explanation of given code in 2-3 lines.
# 2. Detailed functionality including the components the code depends on if any in not more than 100 words.

# """
# prompt = PromptTemplate.from_template(template)
# chain = LLMChain(llm=llm, prompt=prompt)

# docs_to_docgen = "\n\n".join([res["source_code"] for res in expanded_results])
# documentation = chain.invoke({"query_code": query_code, "components": docs_to_docgen})

# print(documentation['text'])
