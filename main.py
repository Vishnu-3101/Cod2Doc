import sys
import logging
import os
from utils.build_graph import BuildGraph

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