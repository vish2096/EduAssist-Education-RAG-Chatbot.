# EduAssist – Education RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for a student helpdesk domain. It retrieves relevant passages from local PDF/TXT documents and uses Groq LLM to generate a grounded answer.

## RAG workflow
1. **Document Loading:** Reads PDF and TXT files from `data/`.
2. **Text Splitting:** Splits documents into chunks (700 characters, 100 overlap).
3. **Embeddings:** Creates vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Database:** Stores vectors in FAISS.
5. **Retriever:** Retrieves the top 4 relevant chunks.
6. **LLM:** Sends retrieved context and the question to Groq (Llama 3.3 70B).
7. **Final Response:** Answers from retrieved context and displays source filenames.

## Requirements
- Python 3.10 or newer
- Groq API key

## Run locally
```bash
# 1. Open this project folder in VS Code
# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# 4. Create .env from the example and add your own key
copy .env.example .env

# 5. Start the chatbot
streamlit run app.py
```
Edit `.env` and replace the placeholder with your Groq API key. Never upload `.env` to GitHub.

## Add your own knowledge
Place additional `.pdf` or `.txt` files inside the `data/` folder, then restart the app. The included `student_guide.txt` is sample general guidance, not an official college policy.

## Push to GitHub
Create a new empty GitHub repository, then run:
```bash
git init
git add .
git commit -m "Build EduAssist education RAG chatbot"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/eduassist-rag-chatbot.git
git push -u origin main
```

## Example questions
- How can I prepare for exams effectively?
- What documents might be needed for a scholarship?
- How can I improve my time management?
- What should I do if I do not understand a topic?

## Important
Scholarship, admission, and fee rules vary. Verify current details with the relevant official institution or government portal. This demo must not be treated as an official policy source.
