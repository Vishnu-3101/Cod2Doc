from langchain_google_genai import ChatGoogleGenerativeAI
from prompts.doc_prompt import doc_prompt

def get_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0
    )
    return doc_prompt | llm