import re
import json

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
        "file_path": graph[entry_point_id]['file_path'],
        "start_line" : graph[entry_point_id]["start_line"],
        "previous_docs": prev_docs,
        "dependent_comps": dependent_comps
    })
    # print("Response from LLM: ", doc.content)
   

    clean_output = re.sub(r"^```(?:json)?\s*|\s*```$", "", doc.content.strip())

    clean_output = re.sub(r'(?<=f)"(.*?)"', r'f\"\1\"', clean_output)

    # 3. Parse the cleaned JSON
    output = json.loads(clean_output) 

    match = output["content"]

    print(match)

    extracted_content = ""

    if match:
        extracted_content = match.strip()
    else:
        print("No answer found.")


    # print("Current Docs: ",extracted_content)

    print(entry_point_id)
    # print(graph[entry_point_id]['source_code'])
    print("--------------------------------------------")

    documentation_parts.append(output)
    conversation_history.append(extracted_content)

    for deps in graph[entry_point_id]['depends_on']:
        generate_docs(deps, graph, chain, seen, ids, documentation_parts, conversation_history)
    
    return documentation_parts