import os
import requests
from dotenv import load_dotenv

# LangChain + Groq
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage

# === Load API keys from .env ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Please set the GROQ_API_KEY environment variable in .env")

# === Step 1: Load and chunk the Nigerian Constitution PDF ===
def load_constitution(pdf_path):
    """Loads and chunks the Nigerian Constitution PDF."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return []

# === Step 2: Embed chunks and save vectorstore ===
def build_vectorstore(chunks):
    """Builds and saves the FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("database")
    return vectorstore

# Tool 2: The Tavily web search tool for online fallback
@tool
def tavily_search(query: str):
    """
    Searches the internet for information using the Tavily API.
    Useful for questions that cannot be answered from the local constitution document.
    """
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": 3
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
        return "\n".join([f"{r['title']}: {r['content']}" for r in results]) if results else "No online info found."
    except requests.exceptions.RequestException as e:
        return f"Error during web search: {e}"

# === Step 3: Create the Agent and Executor ===
def get_agent_executor():
    """Initializes and returns the LangChain Agent Executor with memory."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # or "llama-3.1-8b-instant" for faster responses
        groq_api_key=GROQ_API_KEY
    )
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        vectorstore = FAISS.load_local("database", embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"❌ Error loading vector store. Ensure 'database' directory exists. {e}")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        input_key="query"
    )
    
    @tool
    def pdf_qa(query: str):
        """
        Answers questions by searching a local database of the Nigerian Constitution.
        Useful for questions about the constitution that can be answered from a provided document.
        """
        answer = qa_chain.invoke({"query": query})
        return answer.get("result", "")

    # Define a prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You have two tools: `pdf_qa` for the Constitution and `tavily_search` for web info. Always try pdf_qa first. If insufficient, use tavily_search. Then provide a clear, final answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Initialize tools
    tools = [pdf_qa, tavily_search]
    
    # Create agent + executor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor

# === Console Chat App ===
def main():
    print("🇳🇬 Nigerian Constitution Chatbot (Groq-powered)")
    print("Type 'exit' to quit.\n")

    # Load constitution
    PDF_PATH = "constitution.pdf"
    chunks = load_constitution(PDF_PATH)
    if chunks:
        build_vectorstore(chunks)
        agent_executor = get_agent_executor()
    else:
        return
    
    chat_history = []

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        try:
            # Convert stored history to LangChain format
            langchain_history = []
            for user_msg, ai_msg in chat_history:
                langchain_history.append(HumanMessage(content=user_msg))
                langchain_history.append(AIMessage(content=ai_msg))

            response = agent_executor.invoke({"input": query, "chat_history": langchain_history})
            answer = response["output"]
            print(f"Bot: {answer}\n")

            chat_history.append((query, answer))
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
