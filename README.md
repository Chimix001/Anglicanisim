# Anglican Catechism AI

An AI-powered Christian assistant that helps users explore the Anglican Catechism, ask Bible-related questions, find Bible verses, and receive simple explanations of Scripture.

## Features

* 📖 **Anglican Catechism Q&A**
  Uses Retrieval-Augmented Generation (RAG) to search the Anglican Catechism and provide relevant answers.

* ✝️ **Bible Verse Search**
  Retrieves Bible verses using a public Bible API.

* 💡 **Bible Verse Explanation**
  Provides simple explanations of specific Bible verses.

* 🤖 **AI Agent**
  Uses LangGraph to decide whether to answer directly or use the appropriate tool.

* 🔎 **Topic Classification**
  Filters questions and keeps the assistant focused on Christianity, the Bible, Anglicanism, and related topics.

* ⚡ **FastAPI Backend**
  Provides an API endpoint for interacting with the AI agent.

## Architecture

```text
                     User
                       │
                       ▼
                 FastAPI API
                       │
                       ▼
               LangGraph Agent
                       │
            ┌──────────┼──────────┐
            │          │          │
            ▼          ▼          ▼
       Catechism    Bible      Bible
          RAG       Search    Explanation
            │          │          │
            ▼          ▼          ▼
        ChromaDB   Bible API   Bible API
            │
            ▼
     Anglican Catechism PDF
```

## How It Works

### 1. Anglican Catechism RAG

The Anglican Catechism PDF is loaded and divided into smaller chunks.

The chunks are converted into embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are stored in ChromaDB.

When a user asks a Catechism-related question, the system retrieves the most relevant sections from the vector database and provides them to the AI agent.

### 2. Bible Tools

The agent can retrieve specific Bible verses using the HelloAO Bible API.

A fixed book-name-to-API-ID mapping is used to avoid issues with automatically resolving Bible book names through the API.

For example:

```python
book_map = {
    "Genesis": "GEN",
    "Exodus": "EXO",
    "John": "JHN",
    "Romans": "ROM",
    "Revelation": "REV"
}
```

### 3. LangGraph Agent

LangGraph manages the agent workflow.

The workflow includes:

```text
User Question
      │
      ▼
Question Classifier
      │
      ├── Off-topic ──► Response
      │
      ▼
     Agent
      │
      ├── Catechism Tool
      ├── Bible Search Tool
      ├── Bible Explanation Tool
      │
      ▼
    Response
```

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### AI / LLM

* LangChain
* LangGraph
* Groq
* `openai/gpt-oss-120b`

### RAG

* ChromaDB
* Hugging Face Sentence Transformers
* `sentence-transformers/all-MiniLM-L6-v2`
* PyMuPDF
* LangChain document loaders and text splitters

### Bible

* HelloAO Bible API

### Other

* Pydantic
* python-dotenv
* Requests
* Git/GitHub

## Project Structure

```text
anglican-catechism-ai/
│
├── data/
│   └── anglican_catechism.pdf
│
├── app/
│   └── ...
│
├── RagAgent.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact file structure may change as the project develops.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/anglican-catechism-ai.git
```

Move into the project:

```bash
cd anglican-catechism-ai
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file to GitHub.

The `.gitignore` file excludes:

```text
.env
venv/
__pycache__/
*.pyc
chroma_db/
```

## Running the Agent

You can run the agent directly:

```bash
python RagAgent.py
```

The application will allow you to ask questions from the terminal.

Example:

```text
You: What does the Anglican Catechism say about baptism?

Assistant: ...
```

## Running the FastAPI Backend

Start the API server:

```bash
uvicorn main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Chat Endpoint

```http
POST /chat
```

Example request:

```json
{
  "message": "What does the Catechism say about baptism?"
}
```

Example Bible question:

```json
{
  "message": "What does John 3:16 say?"
}
```

## Example Questions

You can ask questions such as:

```text
What does the Anglican Catechism say about baptism?

What does the Catechism teach about prayer?

What does John 3:16 say?

Explain Romans 6:18.

Give me a Bible verse about forgiveness.

What does Christianity teach about becoming like Christ?
```

## Future Improvements

* 📚 Larger Bible knowledge base for topical Bible searches
* 🔎 Improved Bible reference detection
* 🧠 Better conversation memory
* 🔐 Authentication and user sessions
* ☁️ Cloud deployment
* 📊 Monitoring and evaluation
* 🧪 Automated tests
* 📱 Mobile-friendly interface

## Disclaimer

This project is an experimental AI application built for educational and Christian learning purposes.

AI-generated explanations may contain mistakes. Users should consult the Anglican Catechism, Bible, and qualified Christian teachers or clergy for authoritative theological guidance.

⭐ If you find this project interesting, feel free to explore the code and follow the development.
