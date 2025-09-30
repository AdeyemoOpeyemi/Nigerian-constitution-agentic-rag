import os
import requests
import streamlit as st
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
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# === Load API keys from .env ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()

# === Step 1: Load and chunk the Nigerian Constitution PDF ===
@st.cache_resource
def load_constitution(pdf_path):
    """Loads and chunks the Nigerian Constitution PDF."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)
    except Exception as e:
        st.error(f"❌ Error loading PDF: {e}")
        return []

# === Step 2: Embed chunks and save vectorstore ===
@st.cache_resource
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
@st.cache_resource
def get_agent_executor():
    """Initializes and returns the LangChain Agent Executor with memory."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # You can also try "llama-3.1-8b-instant" (faster/cheaper)
        groq_api_key=GROQ_API_KEY
    )
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        vectorstore = FAISS.load_local("database", embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Error loading vector store. Ensure 'database' directory exists. {e}")
        return None

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

    # Define a prompt that explicitly tells the LLM when to use which tool
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You have two tools: `pdf_qa` to answer questions about the Nigerian Constitution from a local document, and `tavily_search` to find information online. First, always try to use the `pdf_qa` tool. If the answer from `pdf_qa` is not sufficient, is unhelpful, or explicitly states it cannot find the answer, you MUST use the `tavily_search` tool to find a better answer. After a search, always provide a clear, final answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Initialize the tools
    tools = [pdf_qa, tavily_search]
    
    # Create the agent with a chat history placeholder
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the agent executor with memory
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor

# === Streamlit App ===
def main():
    st.title("🇳🇬 Nigerian Constitution Chatbot (Groq-powered)")
    st.markdown("Ask me anything about the Nigerian Constitution! The agent will use a local PDF and can fall back to the web if needed.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load resources
    with st.spinner("Loading constitution and creating agent... This may take a moment."):
        PDF_PATH = "constitution.pdf"
        chunks = load_constitution(PDF_PATH)
        if chunks:
            build_vectorstore(chunks)
            agent_executor = get_agent_executor()
        else:
            st.stop()
    
    # Accept user input
    if query := st.chat_input("What is Chapter 2 of the Nigerian Constitution?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Convert the stored history to LangChain's message format
                    langchain_history = []
                    for user_msg, ai_msg in st.session_state.chat_history:
                        langchain_history.append(HumanMessage(content=user_msg))
                        langchain_history.append(AIMessage(content=ai_msg))

                    response = agent_executor.invoke({"input": query, "chat_history": langchain_history})
                    st.session_state.messages.append({"role": "assistant", "content": response['output']})
                    st.session_state.chat_history.append((query, response['output']))
                    st.markdown(response['output'])
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
