import sys
import logging
import os
import json
from collections import deque

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

def find_entrypoints(graph):

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


def retrieve(entry_point_id):
    expanded = []
    seen = set()
    
    def add_with_deps(comp_id):
        if comp_id in seen or comp_id not in graph:
            return
        seen.add(comp_id)
        expanded.append(graph[comp_id])
        for dep in graph[comp_id]["depends_on"]:
            add_with_deps(dep)

    add_with_deps(entry_point_id)

    return expanded

"""
For every entry point, get their source code and make the doc - line by line...

"""


# for entry_point in entry_points

docs, graph = get_doc(dependency_graph_path)

entry_points = find_entrypoints(graph)

print(entry_points)

# for entry_point_id in entry_points:
expanded_results = retrieve(entry_points[0])
    


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
# results,expanded_results = retrieve_with_dependencies(query_code, hybrid_retriever, graph)


# for result in results:d
#     # print(result.metadata['score'])
#     print(result.page_content)
#     print("------------------------------------")

# print(expanded_results)

for res in expanded_results:
    print(res["id"], "=> depends on", res["depends_on"])
    print("------------------------------------")


# llm = ChatOllama(model="qwen3:0.6b")

# template = """
# You are a code documentation generator who generates developer friendly docuementation.
# Generate documentation for the {query_code}. Use {components} for any additional information 
# required to generate detailed documentation. The docuementation should be a top-down document The documentation should be very detailed and should be structured as described below:

# 1. 1st a brief introduction involving functionality and explanation of code in 2-3 lines.
# 2. It should be followed by detailed explanation of code from top - {query_code} till the leaf dependency components.
# 3. Every code's explanation should be followed by its code.

# """
# prompt = PromptTemplate.from_template(template)
# chain = LLMChain(llm=llm, prompt=prompt)

intro_prompt = PromptTemplate.from_template("""
You are a professional documentation generator. 
Your task is to generate clear, developer-friendly documentation for a software repository. 
The documentation is targeted at new developers onboarding. 

# Instructions:
1. Start with a **concise introduction** (2–4 sentences) of what the entry point does.
2. Then explain the code execution flow **only for the entry point and its immediate dependencies**.
3. For each component:
   - Explain what it does, why it exists, and how it contributes to execution.
   - Show its exact source code in Markdown format:
     ```python
     # code here
     ```
   - After each code block, add `[See code above](#)` as a placeholder link.

# Input:
- Entry Point Code: {query_code}
- Immediate Dependencies: {components}

# Output:
Generate Markdown documentation with an introduction followed by code explanations. 
""")

followup_prompt = PromptTemplate.from_template("""
You are continuing a top-down documentation generation task. 
Below is the documentation generated so far (only last few sections):

{previous_docs}

# Instructions:
1. Continue the documentation without repeating the introduction. 
2. Only explain the new components provided below.
3. Maintain stylistic consistency with the existing documentation.
4. Follow the same format: Explanation → Code snippet → [See code above](#).

# New Components:
{components}

# Output:
Generate Markdown documentation for the new components only.
""")


'''
Instead of giving the compelete components as single file, it is better to add memory and pass components one after the other.
This way it would be more systematic.
'''

llm = ChatOllama(model="qwen3:0.6b")

intro_chain = LLMChain(llm=llm, prompt=intro_prompt)
followup_chain = LLMChain(llm=llm, prompt=followup_prompt)

def generate_docs(entry_point_id, graph, memory_window=3):
    """Generate documentation step by step, expanding dependencies layer by layer,
    with short-term memory of last few sections for consistency.
    """

    seen = set()
    documentation_parts = []
    conversation_history = deque(maxlen=memory_window)  # <-- rolling memory

    def expand_layer(component_ids, is_first=False):
        print(component_ids)
        nonlocal documentation_parts, conversation_history
        new_components = [graph[cid] for cid in component_ids if cid not in seen]
        if not new_components:
            return

        for comp in component_ids:
            seen.add(comp)

        comps_code = "\n\n".join([c["source_code"] for c in new_components])
        print(comps_code)

        if is_first:
            print("")
            # doc = intro_chain.invoke({
            #     "query_code": graph[entry_point_id]["source_code"],
            #     "components": comps_code
            # })
        else:
            prev_docs = "\n\n".join(conversation_history)
            # doc = followup_chain.invoke({
            #     "components": comps_code,
            #     "previous_docs": prev_docs
            # })

        # print(doc['text'])
        print("------------------------------")

        # documentation_parts.append(doc["text"])
        # conversation_history.append(doc["text"])  # keep only last few sections

        # collect dependencies for next layer
        next_layer = []
        for comp in new_components:
            next_layer.extend(comp["depends_on"])
        return next_layer

    # Step 1: entry point + first layer
    first_layer = graph[entry_point_id]["depends_on"]
    next_layer = expand_layer([entry_point_id] + first_layer, is_first=True)

    # Step 2+: iteratively expand
    while next_layer:
        next_layer = expand_layer(next_layer, is_first=False)

    # Combine all partial outputs
    return "\n\n".join(documentation_parts)


# Example usage
docs, graph = get_doc(dependency_graph_path)
entry_points = find_entrypoints(graph)
final_docs = generate_docs(entry_points[0], graph)

with open("output/documentation.md", "w") as f:
    f.write(final_docs)

print("Documentation generated in output/documentation.md")


# docs_to_docgen = "\n\n".join([res["source_code"] for res in expanded_results])
# documentation = chain.invoke({"query_code": graph[entry_points[0]]['source_code'], "components": docs_to_docgen})
# documentation = llm.invoke("hello")

# print(documentation['text'])
