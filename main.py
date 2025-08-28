import sys
from utils.parser import DependencyParser
from utils.toposort import build_graph_from_components, dependency_first_dfs
import logging

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

parser = DependencyParser(repo_path)
components = parser.parse_repository()
    
# Save the dependency graph for future reference
parser.save_dependency_graph(dependency_graph_path)
logger.info(f"Dependency graph saved to: {dependency_graph_path}")

# Build the graph for traversal
graph = build_graph_from_components(components)

# Create a dependency graph in the format expected by the orchestrator
# Dictionary mapping component paths to their dependencies
dependency_graph = {}
for component_id, deps in graph.items():
    dependency_graph[component_id] = list(deps)

# Perform DFS-based traversal
logger.info("Performing DFS traversal on the dependency graph (starting from nodes with no dependencies)")
sorted_components = dependency_first_dfs(graph)
logger.info(f"Sorted {len(sorted_components)} components for processing")

for comp_id in sorted_components:
    print(comp_id)

for i, comp_id in enumerate(sorted_components):
    component = components[comp_id]
    print(f"{i+1}. Processing {component.id} (Type : {component.component_type})")
    print(f"Source : {component.source_code[:50]}...")