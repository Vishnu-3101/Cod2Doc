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
from datetime import datetime


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

st.subheader("📥 Input Repository or Upload File")

option = st.radio(
    "Choose input type:",
    ("🔗 GitHub Repository Link", "📄 Upload Python File")
)


repo_path = None

if option == "🔗 GitHub Repository Link":
    repo_link = st.text_input("Enter GitHub Repository URL:")
    if repo_link:
        try:
            clone_dir_name = repo_link.split("/")[-1].split(".")[0]
            repo_path = f"knowledge_base/{clone_dir_name}"
            if not os.path.exists(repo_path):
                with st.spinner("🔄 Cloning repository..."):
                    clone_repo(repo_url=repo_link, clone_dir=repo_path)
            else:
                progress_placeholder.info("📂 Repository already exists locally.")
        except Exception as e:
            st.error(f"❌ Error cloning repository: {e}")

elif option == "📄 Upload Python File":
    uploaded_file = st.file_uploader("Upload a Python file", type=["py"])
    if uploaded_file is not None:
        uid = datetime.now().strftime('%Y%m%d%H%M%S')
        upload_dir = f"knowledge_base/{uploaded_file.name}_{uid}"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        progress_placeholder.info(f"✅ File uploaded successfully: {uploaded_file.name}")
        repo_path = upload_dir

if st.button("🚀 Generate Documentation") and repo_path:
    # for entry_point in entry_points
    dir_name = repo_path.split("/")[-1]
    dependency_graph_path = f"output/dependency_graph_{dir_name}.json"

    # repo_path = "knowledge_base/PowerPoint-Generator-Python-Project" 

    if not os.path.exists(dependency_graph_path):
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


