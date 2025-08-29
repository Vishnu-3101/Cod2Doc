import json
from langchain.schema import Document

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.runnables import chain


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
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query,k=4))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    best_doc = sorted(docs, key=lambda d: d.metadata["score"],reverse=True)
    return best_doc

# Example
query_code = "back = _backward()"
results,expanded_results = retrieve_with_dependencies(query_code, retriever, data)
# for res in expanded_results:
#     print(res["id"], "=> depends on", res["depends_on"])

# print(results)

for result in results:
    print(result.metadata['score'])
    print(result.page_content)

# print(len(docs))