from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter, RecursiveCharacterTextSplitter


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["depends_on"] = record.get("depends_on")
    metadata["id"] = record.get("id")
    metadata["component_type"] = record.get("component_type")
    metadata["docstring"] = record.get("docstring")
    return metadata


file_path = 'dependency_graph.json'
loader = JSONLoader(
    file_path=file_path,
    jq_schema='.[]',
    # text_content=False,
    content_key="source_code",
    metadata_func=metadata_func
)


splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)


# data = json.loads(Path(file_path).read_text())

data = loader.load()

splits = splitter.split_documents(data)

for chunk in splits[:5]:
    print(chunk.page_content)
    print(chunk.metadata)
    print("----------------------------")

print(len(splits))