import json
from langchain.schema import Document

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.runnables import chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


with open("output/dependency_graph.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []
for comp_id, comp in data.items():
    docs.append(
        Document(
            page_content=comp["source_code"],  # searchable content
            metadata={
                "id": comp["id"],
                "component_type": comp["component_type"],
                "depends_on": comp["depends_on"],
                "file_path": comp["file_path"],
                "relative_path": comp["relative_path"],
            }
        )
    )


def retrieve_with_dependencies(query, retriever, data):
    results = retriever.invoke(query)
    
    expanded = []
    seen = set()
    
    def add_with_deps(comp_id):
        if comp_id in seen or comp_id not in data:
            return
        seen.add(comp_id)
        expanded.append(data[comp_id])
        for dep in data[comp_id]["depends_on"]:
            add_with_deps(dep)

    for doc in results:
        add_with_deps(doc.metadata["id"])
    
    return results,expanded


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

# embeddings = OllamaEmbeddings(
#     model="qwen3:0.6b",
# )

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

# @chain
# def retriever(query: str) -> List[Document]:
#     docs, scores = zip(*vectorstore.similarity_search_with_score(query,k=4))
#     for doc, score in zip(docs, scores):
#         doc.metadata["score"] = score

#     best_doc = sorted(docs, key=lambda d: d.metadata["score"],reverse=True)
#     return best_doc

# Example
query_code = "def backward():"
results,expanded_results = retrieve_with_dependencies(query_code, hybrid_retriever, data)
# for res in expanded_results:
#     print(res["id"], "=> depends on", res["depends_on"])

# print(results)

for result in results:
    # print(result.metadata['score'])
    print(result.page_content)
    print("------------------------------------")

# print(len(docs))