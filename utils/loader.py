import json
from langchain.schema import Document
from typing import List
from langchain_core.runnables import chain


def get_doc(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
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
    return docs,data


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

