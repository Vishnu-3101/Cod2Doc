def retrieve(graph,entry_point_id):
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