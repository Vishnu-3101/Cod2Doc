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