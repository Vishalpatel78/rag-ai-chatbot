import streamlit as st
import tempfile, os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()  # loads GOOGLE_API_KEY and GROQ_API_KEY


# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)


# ============================================================
# PDF PROCESSING (the RAG SETUP phase)
# ============================================================

# @st.cache_resource = run this ONCE per file, not on every message
@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes, file_name):
    """Takes uploaded PDF -> returns a ready retriever."""

    # Save uploaded file temporarily (PyPDFLoader needs a real path)
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # STEP 1: Load the PDF
    docs = PyPDFLoader(tmp_path).load()

    # STEP 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # STEP 3: Create embeddings (free Gemini)
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # STEP 4: Store in Chroma (in-memory - fresh for each PDF)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    os.remove(tmp_path)  # clean up the temp file

    # Create the retriever (MMR = relevant + diverse chunks)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return retriever, len(chunks)


# ============================================================
# THE LLM + PROMPT
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)

# Prompt engineering: answer ONLY from context + format nicely
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions about a document.\n\n"
        "RULES:\n"
        "- Answer using ONLY the context below.\n"
        "- If the answer isn't in the context, say: "
        "\"I couldn't find that in the document.\"\n"
        "- Write clean, well-structured answers using short paragraphs "
        "or bullet points where helpful.\n"
        "- Do NOT mention chunks, pages, context, or sources.\n"
        "- Never make up information.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])


# ============================================================
# GENERATE ANSWER
# ============================================================

def get_answer(retriever, question):
    """Retrieve relevant chunks + generate a clean answer."""

    # RETRIEVAL
    docs = retriever.invoke(question)

    # AUGMENTATION - build context
    context = "\n\n".join(
        d.page_content for d in docs
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # GENERATION
    response = llm.invoke(messages)

    # Gemini may return content as a string or a list of blocks
    if isinstance(response.content, str):
        return response.content

    return "".join(
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict)
    )


# ============================================================
# SIDEBAR - PDF Upload
# ============================================================

with st.sidebar:

    st.header("📄 Your Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf"
    )

    if uploaded_file:

        # Process only if it's a new file
        if st.session_state.get("current_file") != uploaded_file.name:

            with st.spinner("Reading your PDF..."):
                    retriever, n_chunks = process_pdf(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )

            st.session_state.retriever = retriever
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # reset chat for new PDF

            st.success(f"Ready! ({n_chunks} sections indexed)")

        else:
            st.success("Document loaded ✓")

        st.caption(f"File: {uploaded_file.name}")

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================

st.title("💬 Chat with your PDF")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# If no PDF yet, show a friendly prompt
if "retriever" not in st.session_state:
    st.info("📄 Upload a PDF from the sidebar to get started.")
    st.stop()


# Show previous messages (like ChatGPT)
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input at the bottom
if question := st.chat_input("Ask anything about your document..."):

    # Show user's message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # Generate and show the answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = get_answer(
                st.session_state.retriever,
                question
            )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# streamlit run app.py