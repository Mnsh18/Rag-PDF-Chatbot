import streamlit as st
import time

st.set_page_config(
    page_title="Talk with your Pdf",
    page_icon="📚",
    layout="centered"
)

st.markdown("""
<style>
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #888;
        border-radius: 15px;
        padding: 8px;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("Chat with your PDF")

import pymupdf
from sentence_transformers import SentenceTransformer
import faiss


st.subheader("📄 Upload your PDF")
st.caption("Upload a PDF and ask questions about its contents.")

uploaded_file = st.file_uploader(
    "Select PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

st.divider()


# Sidebar
with st.sidebar:
    st.title("📚 RAG Application")

    st.divider()

    st.subheader("💡 How it works")

    st.write("📄 **Upload**")
    st.caption("Choose a PDF document.")

    st.write("🔍 **Retrieve**")
    st.caption("Find relevant information from your PDF.")

    st.write("🤖 **Ask**")
    st.caption("Get an AI-generated answer.")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


if uploaded_file:

    # Upload success message
    if uploaded_file.name != st.session_state.uploaded_file_name:

        message = st.empty()

        message.success("PDF uploaded successfully!")

        time.sleep(2)

        message.empty()

        st.session_state.uploaded_file_name = uploaded_file.name


    # PDF extraction
    pdf_bytes = uploaded_file.read()

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        text = text + page.get_text()


    # Chunking
    chunks = []

    for i in range(0, len(text), 500):

        chunk = text[i:i + 500]

        chunks.append(chunk)


    # Embeddings
    model = load_model()

    embeddings = model.encode(chunks)


    # FAISS
    dimensions = len(embeddings[0])

    index = faiss.IndexFlatL2(dimensions)

    index.add(embeddings)


    # Question input
    question = st.chat_input(
        "Ask a question about your PDF"
    )


    if question:

        # Question embedding
        start = time.time()

        question_embedding = model.encode([question])

        st.write(
            "Embedding time:",
            time.time() - start
        )


        # FAISS retrieval
        start = time.time()

        distances, indices = index.search(
            question_embedding,
            1
        )

        st.write(
            "FAISS time:",
            time.time() - start
        )


        # Build context
        context = ""

        for i in indices[0]:

            context = context + chunks[i]


        # Prompt
        prompt = """Answer the question using only the provided context.

If the answer is not present in the context, say:
"The answer is not available in the provided PDF."

Do not use outside knowledge or make up an answer.

Context:
""" + context + "\nQuestion: " + question


        # Gemini
        from google import genai

        client = genai.Client()

        start = time.time()


        stream = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt,
            generation_config={
                "thinking_level": "minimal"
            },
            stream=True
        )


        # Collect streamed answer
        answer = ""

        for event in stream:

            if event.event_type == "step.delta":

                if event.delta.type == "text":

                    answer += event.delta.text


        # Gemini timing
        st.write(
            "Gemini time:",
            time.time() - start
        )


        # Save conversation
        st.session_state.chat_history.append(
            (question, answer)
        )


# Display chat history
for question, answer in st.session_state.chat_history:

    # User message - right side
    col1, col2 = st.columns([1, 2])

    with col2:
        st.info(question)


    # Assistant message - left side
    col1, col2 = st.columns([2, 1])

    with col1:
        st.success(answer)