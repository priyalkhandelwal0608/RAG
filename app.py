import streamlit as st
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #1e1e2f, #2b2b45);
    color: white;
}

h1 {
    text-align: center;
    color: white;
}

.upload-box {
    border: 2px dashed #6c63ff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    background: rgba(255,255,255,0.05);
    margin-bottom: 20px;
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

.stTextInput input {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    "<h1>🤖 RAG Document Chatbot</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Upload a TXT file and chat with it 🚀</p>",
    unsafe_allow_html=True
)

# -----------------------------
# Upload UI
# -----------------------------
st.markdown(
    "<div class='upload-box'>📂 Upload your .txt document</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload TXT File",
    type=["txt"],
    label_visibility="collapsed"
)

# -----------------------------
# Chat History
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# File Processing
# -----------------------------
if uploaded_file is not None:

    # Read file
    text = uploaded_file.read().decode("utf-8")

    # -----------------------------
    # Text Splitter
    # -----------------------------
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.split_text(text)

    # -----------------------------
    # Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # FAISS Vector Store
    # -----------------------------
    vectorstore = FAISS.from_texts(docs, embeddings)

    # -----------------------------
    # Ollama LLM
    # -----------------------------
    llm = Ollama(model="llama3")

    st.success("✅ File uploaded and processed successfully!")

    # -----------------------------
    # User Prompt
    # -----------------------------
    prompt = st.text_input(
        "💬 Ask something about your document..."
    )

    # -----------------------------
    # Send Button
    # -----------------------------
    if st.button("🚀 Send") and prompt:

        # -----------------------------
        # Similarity Search
        # -----------------------------
        docs_with_scores = vectorstore.similarity_search_with_score(
            prompt,
            k=3
        )

        # -----------------------------
        # Filter Relevant Docs
        # -----------------------------
        relevant_docs = [
            doc for doc, score in docs_with_scores
            if score < 1.5
        ]

        # -----------------------------
        # No Relevant Info
        # -----------------------------
        if len(relevant_docs) == 0:

            response = "❌ Information not found in the uploaded document."

        else:

            # Combine Context
            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )

            # -----------------------------
            # Final Prompt
            # -----------------------------
            final_prompt = f"""
You are a strict AI assistant.

Rules:
- Answer ONLY from the given context
- If answer is not present, say:
  "Information not found in document"
- Do NOT hallucinate
- Keep answers concise

Context:
{context}

User Question:
{prompt}

Answer:
"""

            # Generate Response
            response = llm.invoke(final_prompt)

            # Limit Response Length
            response = " ".join(response.split()[:200])

        # -----------------------------
        # Save Chat
        # -----------------------------
        st.session_state.chat_history.append(
            ("user", prompt)
        )

        st.session_state.chat_history.append(
            ("ai", response)
        )

# -----------------------------
# Display Chat
# -----------------------------
for role, message in st.session_state.chat_history:

    if role == "user":

        st.markdown(
            f"<div class='chat-user'>🧑 {message}</div>",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"<div class='chat-ai'>🤖 {message}</div>",
            unsafe_allow_html=True
        )
