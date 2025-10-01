import re

def generate_docs(entry_point_id, graph, chain, seen, ids, documentation_parts, conversation_history):
    """Generate documentation step by step, expanding dependencies layer by layer,
    with short-term memory of last few sections for consistency.
    """
    if entry_point_id in seen or entry_point_id not in ids:
        return
    seen.append(entry_point_id)

    prev_docs = "\n\n".join([c for c in conversation_history])
    # print("PrevDocs:", prev_docs) 
    dependent_comps = graph[entry_point_id]['depends_on']

    doc = chain.invoke({
        "query_code": graph[entry_point_id]["source_code"],
        "previous_docs": prev_docs,
        "dependent_comps": dependent_comps
    })
    # print("Response from LLM: ", doc.content)

    match = re.search(r"<answer>(.*?)</answer>", doc.content, re.DOTALL)

    extracted_content = ""

    if match:
        extracted_content = match.group(1).strip()
    else:
        print("No <answer> tags found.")


    # print("Current Docs: ",extracted_content)

    print(entry_point_id)
    # print(graph[entry_point_id]['source_code'])
    print("--------------------------------------------")

    documentation_parts.append(extracted_content)
    conversation_history.append(extracted_content)

    for deps in graph[entry_point_id]['depends_on']:
        generate_docs(deps, graph, chain, seen, ids, documentation_parts, conversation_history)
    
    return "\n----------------------\n".join(documentation_parts)