import sys
import logging
import os
from utils.build_graph import BuildGraph
from utils.loader import retrieve_with_dependencies, get_doc
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("docstring_generator")


dependency_graph_path = "output/dependency_graph.json"
repo_path = "knowledge_base"

if not os.path.exists(dependency_graph_path):
    BuildGraph(repo_path=repo_path, dependency_graph_path=dependency_graph_path)

else:
    print("Dependency graph already created. Using it for further processing....")


docs, data = get_doc(dependency_graph_path)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

vectorstore = FAISS.from_documents(docs, embeddings)
embedding_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

# Hybrid retriever (weighted ensemble)
hybrid_retriever = EnsembleRetriever(
    retrievers=[embedding_retriever, bm25_retriever],
    weights=[0.3, 0.7] 
)

# Example
query_code = "def backward():"
results,expanded_results = retrieve_with_dependencies(query_code, hybrid_retriever, data)


for result in results:
    # print(result.metadata['score'])
    print(result.page_content)
    print("------------------------------------")

# print(expanded_results)

for res in expanded_results:
    print(res["id"], "=> depends on", res["depends_on"])
    print("------------------------------------")