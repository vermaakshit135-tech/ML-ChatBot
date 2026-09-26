"""
MLChatBot — RAG Chatbot UI
Streamlit interface wrapping the LangChain + Mistral AI RAG pipeline.

Setup:
    pip install streamlit
Run:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="MLChatBot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Theme — black background, gold as the dominant accent, embossed 3D surfaces
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --gold-light: #f5da8a;
    --gold: #d4af37;
    --gold-dark: #8a6d1f;
    --ink: #050505;
    --panel: #121212;
    --text: #f2ede0;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background:
        radial-gradient(ellipse at top, #161616 0%, var(--ink) 55%),
        var(--ink);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

.block-container { max-width: 720px; padding-top: 2.5rem; }

/* ---- Header ---- */
.mlc-title {
    font-family: 'Cinzel', serif;
    font-weight: 700;
    font-size: 2.5rem;
    text-align: center;
    margin: 0;
    background: linear-gradient(180deg, var(--gold-light) 15%, var(--gold) 50%, var(--gold-dark) 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 3px rgba(0,0,0,0.6));
    animation: mlc-rise 0.6s ease-out;
}
.mlc-tagline {
    text-align: center;
    color: #a08a52;
    font-size: 0.92rem;
    margin-top: 0.35rem;
    margin-bottom: 1.6rem;
}
@keyframes mlc-rise {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

hr.mlc-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dark), var(--gold), var(--gold-dark), transparent);
    margin-bottom: 1.8rem;
}

/* ---- Chat bubbles ---- */
[data-testid="stChatMessage"] {
    background: linear-gradient(160deg, #171717, #0c0c0c);
    border: 1px solid #2a2210;
    border-radius: 16px;
    box-shadow:
        0 6px 14px rgba(0,0,0,0.55),
        inset 0 1px 0 rgba(212,175,55,0.10);
    margin-bottom: 0.9rem;
}

/* ---- Avatars ("icons") — golden, embossed ---- */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(145deg, var(--gold-light), var(--gold) 55%, var(--gold-dark)) !important;
    box-shadow:
        0 2px 5px rgba(0,0,0,0.5),
        inset 0 1px 1px rgba(255,255,255,0.55),
        inset 0 -2px 3px rgba(0,0,0,0.25);
    border: 1px solid var(--gold-dark);
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    background: var(--panel);
    border: 1px solid var(--gold-dark);
    border-radius: 14px;
    box-shadow: 0 0 0 1px rgba(212,175,55,0.08), 0 4px 16px rgba(0,0,0,0.5);
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }

[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(145deg, var(--gold-light), var(--gold-dark)) !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(212,175,55,0.35);
}
[data-testid="stChatInputSubmitButton"] svg { fill: var(--ink) !important; }

/* ---- Generic icon-ish svgs (spinner, buttons) ---- */
svg { color: var(--gold); }

/* ---- Small utility button (clear chat) ---- */
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--gold-dark) !important;
    color: var(--gold) !important;
    border-radius: 10px !important;
}
button[kind="secondary"]:hover {
    border-color: var(--gold) !important;
    box-shadow: 0 0 8px rgba(212,175,55,0.3);
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--gold-dark); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
title_col, clear_col = st.columns([5, 1])
with title_col:
    st.markdown('<p class="mlc-title">MLChatBot</p>', unsafe_allow_html=True)
with clear_col:
    st.write("")
    if st.button("Clear", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<p class="mlc-tagline">Ask a question about your documents</p>', unsafe_allow_html=True)
st.markdown('<hr class="mlc-divider">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# RAG pipeline — built once, cached across reruns
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading MLChatBot...")
def load_pipeline():
    embedding_model = HuggingFaceEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    llm = ChatMistralAI(model="ministral-3b-latest")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )

    return retriever, llm, prompt


retriever, llm, prompt = load_pipeline()

# ----------------------------------------------------------------------------
# Chat state + history
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# Handle new input
# ----------------------------------------------------------------------------
query = st.chat_input("Message MLChatBot...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            final_prompt = prompt.invoke({
                "context": context,
                "question": query,
            })

            response = llm.invoke(final_prompt)
            st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})