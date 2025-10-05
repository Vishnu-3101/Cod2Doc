import sys
import logging
import os
import shutil
from dotenv import load_dotenv
from utils.build_graph import BuildGraph
from utils.loader import get_doc
from docgen.entrypoints import find_entrypoints
from docgen.retriever import retrieve
from docgen.generator import generate_docs
from llm.chain_setup import get_chain
import streamlit as st
from pathlib import Path
from git import Repo


def clone_repo(repo_url, clone_dir="/knowledge_base/dummy"):
    # Clean up old repo if it exists
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    
    st.info("📦 Cloning repository...")
    Repo.clone_from(repo_url, clone_dir)
    st.success("✅ Repository cloned successfully!")
    return clone_dir




st.set_page_config(
    page_title="AI Doc Generator",
    page_icon="📘",
    layout="wide"
)

st.title("📘 AI Code Documentation Generator")

# Input for repo link
repo_link = st.text_input("🔗 Enter GitHub Repository Link")

# Placeholder for progress messages
progress_placeholder = st.empty()

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

if st.button("🚀 Generate Documentation") and repo_link:
    # for entry_point in entry_points
    dependency_graph_path = "output/dependency_graph.json"
    # repo_path = "knowledge_base/PowerPoint-Generator-Python-Project"
    clone_dir_name = repo_link.split("/")[-1].split(".")[0]
    repo_path = f"knowledge_base/{clone_dir_name}"

    if not os.path.exists(repo_path):
        clone_repo(repo_url=repo_link, clone_dir=repo_path)

    force_build = True

    if not os.path.exists(dependency_graph_path) or force_build:
        progress_placeholder.info("📂 Understanding repository structure...")
        BuildGraph(repo_path=repo_path, dependency_graph_path=dependency_graph_path)

    else:
        progress_placeholder.info("📂 Repository graph already available.")

    progress_placeholder.info("🔍 Finding entry points...")

    docs, graph = get_doc(dependency_graph_path)
    entry_points = find_entrypoints(graph)
    logger.info(f"Entrypoints found: {entry_points}")


    for entry_point in entry_points:

        # for entry_point_id in entry_points:
        expanded_results = retrieve(graph,entry_point)

        for res in expanded_results:
            print(res["id"], "=> depends on", res["depends_on"])
            print("------------------------------------")

        progress_placeholder.info("📝 Generating documentation...")

        llm_chain = get_chain()

        seen = []
        documentation_parts = []
        conversation_history = []

        intro_block = f"""
        `
        {entry_point}
        `

        This is the entry point of the code. The detailed explanation is provided below.
        """

        documentation_parts.append(intro_block)

        final_docs = generate_docs(entry_point, graph, llm_chain,
            seen, list(graph.keys()), documentation_parts, conversation_history
            )
        
        safe_name = entry_point.replace(".", "_").replace(" ", "_")

        progress_placeholder.success(f"✅ Documentation ready for {entry_point}")

        output_file = Path(f"output/documentation_{safe_name}.md")

        with open(output_file, "w", encoding="utf-8", errors="ignore") as f:
            f.write(final_docs)

        print("Documentation generated in output/documentation.md")

        with open(output_file, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {entry_point} Documentation",
                    data=f,
                    file_name=f"documentation_{safe_name}.md",
                    mime="text/markdown"
                )
        
        st.markdown(final_docs, unsafe_allow_html=True)


