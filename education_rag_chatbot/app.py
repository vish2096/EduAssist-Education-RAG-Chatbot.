import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
st.set_page_config(page_title="EduAssist RAG Chatbot", page_icon="🎓", layout="wide")
st.title("🎓 EduAssist – Student Helpdesk RAG Chatbot")
st.caption("Ask questions about study skills, admissions preparation, scholarships, and student support.")

@st.cache_resource
def build_vectorstore():
    docs = []
    data_dir = "data"
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if name.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif name.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    if not docs:
        raise ValueError("No documents found. Add PDF or TXT files to the data folder.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

with st.sidebar:
    st.header("⚙️ Setup")
    st.write("1. Add your learning documents to `data/`.")
    st.write("2. Add your Groq API key to `.env`.")
    st.write("3. Restart the app after changing documents.")
    st.info("This starter project uses the sample student guide included in the data folder.")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.warning("Add GROQ_API_KEY to your .env file, then restart the app.")
    st.code("GROQ_API_KEY=your_groq_api_key_here")
    st.stop()

try:
    with st.spinner("Loading documents and creating the searchable knowledge base..."):
        vectorstore = build_vectorstore()
except Exception as e:
    st.error(f"Could not prepare documents: {e}")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=api_key)
prompt = ChatPromptTemplate.from_template("""You are EduAssist, a helpful student-support assistant.
Answer using ONLY the context below. If the answer is not in the context, say:
"I couldn't find that information in the uploaded student documents."
Do not invent fees, dates, eligibility rules, or official policies.
Keep answers clear and student-friendly.

Context:
{context}

Question: {question}
Answer:""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about student support...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the student knowledge base..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            answer = (prompt | llm).invoke({"context": context, "question": question}).content
            st.markdown(answer)
            sources = sorted({d.metadata.get("source", "Knowledge base") for d in docs})
            with st.expander("📄 Sources used"):
                for source in sources:
                    st.write(os.path.basename(source))
    st.session_state.messages.append({"role": "assistant", "content": answer})
