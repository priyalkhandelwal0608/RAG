import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# -----------------------------
# Custom CSS (UI)
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e1e2f, #2b2b45);
    color: white;
}

h1 {
    text-align: center;
}

.upload-box {
    border: 2px dashed #6c63ff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    background: rgba(255,255,255,0.05);
}

.chat-user {
    background: #6c63ff;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
    color: white;
    width: fit-content;
    max-width: 70%;
}

.chat-ai {
    background: #2f2f4f;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
    color: white;
    width: fit-content;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1>🤖 RAG Document Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a file and chat with it 🚀</p>", unsafe_allow_html=True)

# -----------------------------
# Upload Section
# -----------------------------
st.markdown("<div class='upload-box'>📂 Upload your .txt file</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="txt")

if uploaded_file:

    text = uploaded_file.read().decode("utf-8")

    # -----------------------------
    # Split Text
    # -----------------------------
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_text(text)

    # -----------------------------
    # Embeddings + FAISS
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(docs, embeddings)

    # -----------------------------
    # Load LLM (Ollama)
    # -----------------------------
    llm = Ollama(model="llama3")

    st.success("✅ File uploaded & processed!")

    # -----------------------------
    # Chat History
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -----------------------------
    # Prompt Input
    # -----------------------------
    prompt = st.text_input("💬 Ask something about your document...")

    if st.button("🚀 Send") and prompt:

        # -----------------------------
        # Similarity Search with Score
        # -----------------------------
        docs_with_scores = vectorstore.similarity_search_with_score(prompt, k=3)

        # Filter relevant docs
        relevant_docs = [doc for doc, score in docs_with_scores if score < 0.7]

        # -----------------------------
        # If nothing relevant → No hallucination
        # -----------------------------
        if len(relevant_docs) == 0:
            response = "❌ Information not found in the uploaded document."
        else:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            final_prompt = f"""
You are a strict AI assistant.

Rules:
- Answer ONLY from the context
- If answer is not clearly present, say "Information not found"
- Do NOT guess or add extra info

Context:
{context}

User:
{prompt}

Answer:
"""

            response = llm(final_prompt)

            # Limit length
            response = " ".join(response.split()[:200])

        # Save chat
        st.session_state.chat_history.append(("user", prompt))
        st.session_state.chat_history.append(("ai", response))

    # -----------------------------
    # Chat Display (Bubbles)
    # -----------------------------
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>🤖 {msg}</div>", unsafe_allow_html=True)
