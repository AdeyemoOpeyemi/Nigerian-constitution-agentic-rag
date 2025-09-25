import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# LangChain imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage

# === Load API keys ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Please set GEMINI_API_KEY in your .env file.")
    exit()

# --- Load and chunk PDF ---
def load_constitution(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# --- Build vectorstore ---
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("database")
    return vectorstore

# --- Tavily search ---
@tool
def tavily_search(query: str):
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {"api_key": TAVILY_API_KEY, "query": query, "max_results": 3}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
        return "\n".join([f"{r['title']}: {r['content']}" for r in results]) if results else "No info found."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# --- Create Agent Executor ---
def get_agent_executor():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.load_local("database", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        return None
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        input_key="query"
    )
    @tool
    def pdf_qa(query: str):
        return qa_chain.invoke({"query": query}).get("result", "")
    tools = [pdf_qa, tavily_search]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use pdf_qa first; then tavily_search if needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Main Program ---
def main():
    print("🇳🇬 Nigerian Constitution Chatbot (type 'exit' to quit)\n")
    PDF_PATH = "constitution.pdf"
    chunks = load_constitution(PDF_PATH)
    build_vectorstore(chunks)
    agent_executor = get_agent_executor()
    if not agent_executor:
        return

    chat_history = []

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "exit":
            print("\n✅ Chat ended. Goodbye!")
            break
        # Convert stored chat history to LangChain messages
        langchain_history = []
        for u_msg, a_msg in chat_history:
            langchain_history.append(HumanMessage(content=u_msg))
            langchain_history.append(AIMessage(content=a_msg))
        try:
            response = agent_executor.invoke({"input": query, "chat_history": langchain_history})
            answer = response.get("output", "")
            print(f"Assistant: {answer}")
            chat_history.append((query, answer))
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
