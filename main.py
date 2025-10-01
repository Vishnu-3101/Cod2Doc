import sys
import logging
import os
from dotenv import load_dotenv
from utils.build_graph import BuildGraph
from utils.loader import get_doc
from docgen.entrypoints import find_entrypoints
from docgen.retriever import retrieve
from docgen.generator import generate_docs
from llm.chain_setup import get_chain

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("docstring_generator")

load_dotenv()


"""
For every entry point, get their source code and make the doc - line by line...

"""

# for entry_point in entry_points
dependency_graph_path = "output/dependency_graph.json"
repo_path = "knowledge_base/AutoDiff"

force_build = False

if not os.path.exists(dependency_graph_path) or force_build:
    BuildGraph(repo_path=repo_path, dependency_graph_path=dependency_graph_path)

else:
    print("Dependency graph already created. Using it for further processing....")

docs, graph = get_doc(dependency_graph_path)
entry_points = find_entrypoints(graph)
logger.info(f"Entrypoints found: {entry_points}")

# for entry_point_id in entry_points:
expanded_results = retrieve(graph,entry_points[0])

for res in expanded_results:
    print(res["id"], "=> depends on", res["depends_on"])
    print("------------------------------------")


llm_chain = get_chain()

seen = []
documentation_parts = []
conversation_history = []

intro_block = f"""
`
{entry_points[0]}
`

This is the entry point of the code. The detailed explanation is provided below.
"""

documentation_parts.append(intro_block)

final_docs = generate_docs(entry_points[0], graph, llm_chain,
    seen, list(graph.keys()), documentation_parts, conversation_history
    )

with open("output/documentation.md", "w") as f:
    f.write(final_docs)

print("Documentation generated in output/documentation.md")


