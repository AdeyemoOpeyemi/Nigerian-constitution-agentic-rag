# 🇳🇬 Nigerian Constitution Agentic RAG Chatbot

This project is an **Agentic Retrieval-Augmented Generation (RAG)** chatbot powered by **LangChain + Gemini + FAISS + Tavily**, designed to answer questions about the **Nigerian Constitution**.

Unlike traditional RAG systems, this chatbot uses an **agent with multiple tools** (`pdf_qa` + `tavily_search`) and **conversational memory**, making it context-aware and more robust.

---

##  Features
-  **RAG with Local PDF** → Loads, chunks, and embeds the Nigerian Constitution using **HuggingFace embeddings + FAISS**.
-  **Agentic Workflow** → Uses **LangChain Agent** with multiple tools:
  - `pdf_qa` → Queries the local FAISS database of the Constitution.
  - `tavily_search` → Falls back to Tavily Web Search if the local PDF cannot answer.
-  **Hybrid Knowledge Source** → Combines **offline RAG** with **online retrieval**.
-  **Conversational Memory** → Maintains context across turns for coherent multi-turn conversations.
-  **Streamlit UI** → Simple, interactive chat interface.

---


##  Setup Instructions

###  Clone Repo
```bash
git clone https://github.com/yourusername/nigerian-constitution-agentic-rag.git
cd nigerian-constitution-agentic-rag
