# IMPORT DEPENDENCIES

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

import os
import requests

# Use a fixed book-name-to-API-ID mapping to avoid API book lookup issues
book_map = {
        "Genesis": "GEN",
        "Exodus": "EXO",
        "Leviticus": "LEV",
        "Numbers": "NUM",
        "Deuteronomy": "DEU",
        "Joshua": "JOS",
        "Judges": "JDG",
        "Ruth": "RUT",
        "1 Samuel": "1SA",
        "2 Samuel": "2SA",
        "1 Kings": "1KI",
        "2 Kings": "2KI",
        "1 Chronicles": "1CH",
        "2 Chronicles": "2CH",
        "Ezra": "EZR",
        "Nehemiah": "NEH",
        "Esther": "EST",
        "Job": "JOB",
        "Psalms": "PSA",
        "Proverbs": "PRO",
        "Ecclesiastes": "ECC",
        "Song of Solomon": "SNG",
        "Isaiah": "ISA",
        "Jeremiah": "JER",
        "Lamentations": "LAM",
        "Ezekiel": "EZK",
        "Daniel": "DAN",
        "Hosea": "HOS",
        "Joel": "JOL",
        "Amos": "AMO",
        "Obadiah": "OBA",
        "Jonah": "JON",
        "Micah": "MIC",
        "Nahum": "NAM",
        "Habakkuk": "HAB",
        "Zephaniah": "ZEP",
        "Haggai": "HAG",
        "Zechariah": "ZEC",
        "Malachi": "MAL",
        "Matthew": "MAT",
        "Mark": "MRK",
        "Luke": "LUK",
        "John": "JHN",
        "Acts": "ACT",
        "Romans": "ROM",
        "1 Corinthians": "1CO",
        "2 Corinthians": "2CO",
        "Galatians": "GAL",
        "Ephesians": "EPH",
        "Philippians": "PHP",
        "Colossians": "COL",
        "1 Thessalonians": "1TH",
        "2 Thessalonians": "2TH",
        "1 Timothy": "1TI",
        "2 Timothy": "2TI",
        "Titus": "TIT",
        "Philemon": "PHM",
        "Hebrews": "HEB",
        "James": "JAS",
        "1 Peter": "1PE",
        "2 Peter": "2PE",
        "1 John": "1JN",
        "2 John": "2JN",
        "3 John": "3JN",
        "Jude": "JUD",
        "Revelation": "REV",
    }


# ENVIRONMENT VARIABLES

load_dotenv()

BIBLE_API_KEY = os.getenv("BIBLE_API_KEY")


# LLM


llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


# EMBEDDING MODEL

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# CHROMA DATABASE

db = Chroma(
    collection_name="anglican_catechism",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# LOAD CATECHISM PDF


loader = PyPDFLoader(
    file_path="To-Be-a-Christian.pdf"
)

documents = loader.load()

print(f"PDF LOADED. PAGES: {len(documents)}")


# SPLIT PDF INTO CHUNKS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = text_splitter.split_documents(documents)

print(f"CREATED {len(chunks)} CHUNKS")


# ADD DOCUMENTS ONLY IF
# CHROMA IS EMPTY

existing_documents = db.get(limit=1)

if not existing_documents["ids"]:

    print("Chroma database is empty.")
    print("Adding Catechism documents...")

    db.add_documents(chunks)

    print("Documents added successfully!")

else:

    print("Chroma database already contains documents.")
    print("Skipping document ingestion.")


# RETRIEVER
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# AGENT STATE

class AgentState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    on_topic: bool

    context: str

# QUESTION CLASSIFICATION

class QuestionClassification(BaseModel):

    is_Anglican_related: bool = Field(
        description=(
            "True if the question is related to "
            "Anglicanism, Christianity, Christian living, "
            "Jesus Christ, the Gospel, the Bible, "
            "Bible verses, Christian doctrine, or the "
            "Anglican Catechism."
        )
    )


classifier_llm = llm.with_structured_output(
    QuestionClassification,
    method="function_calling"
)


def question_classifier(state: AgentState):

    question = state["messages"][-1].content

    result = classifier_llm.invoke(
        f"""
        Determine whether the user's question is related to
        any of the following topics:

        - Anglicanism
        - Anglican Catechism
        - Christianity
        - Christian living
        - Jesus Christ
        - Gospel
        - Bible
        - Bible verses
        - Bible references
        - Christian prayer
        - Christian doctrine
        - The Nicene Creed
        - The Apostles' Creed
        - The Creed of Saint Athanasius
        - Becoming like Christ
        - Catechesis
        - Prayers for use with the Catechism
        - A Rite for Admission of Catechumens
        - Articles of Religion

        IMPORTANT:

        Questions about the Bible, Bible verses,
        Bible references, or Christianity are ON-TOPIC
        even when Anglicanism is not explicitly mentioned.

        Question:
        {question}
        """
    )

    return {
        "on_topic": result.is_Anglican_related
    }


# BIBLE VERSE TOOL

@tool
def search_bible_verse(book: str,chapter: int,verse: int):
    """Get a Bible verse by book, chapter, and verse."""

    book_id = book_map.get(book)
    if not book_id:
        return f"unkonown Bible Bokk: {book}"

    url =  (
        f"https://bible.helloao.org/"
        f"api/eng_asv/{book_id}/{chapter}.json"
    )

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Find the requested verse
    for item in data["chapter"]["content"]:

        if item.get("verse") == verse:

            return item.get("text", "")

    return (f"Verse {book} {chapter}:{verse} was not found.")


# BIBLE VERSE EXPLANATION
@tool
def explain_bible_verse(book: str, chapter: int, verse: int):
    """Explain a Bible verse clearly and simply."""

    book_id = book_map.get(book)

    if not book_id:
        return f"Unknown Bible book: {book}"

    url = (
        f"https://bible.helloao.org/"
        f"api/eng_asv/{book_id}/{chapter}.json"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    verse_text = None

    for item in data["chapter"]["content"]:

        if item.get("verse") == verse:
            verse_text = item.get("text", "")
            break

    if not verse_text:
        return f"Verse {book} {chapter}:{verse} was not found."

    prompt = f"""
    Explain this Bible verse clearly and simply.

    Reference: {book} {chapter}:{verse}

    Verse:
    {verse_text}

    Explain:
    1. What the verse means
    2. Its main teaching
    3. Its context, if known

    Do not invent historical or theological claims.
    """

    explanation = llm.invoke(prompt)

    return explanation.content


# CATECHISM SEARCH TOOL

@tool
def search_catechism(question: str):
    """Search the Anglican Catechism for information relevant to the user's question."""

    documents = retriever.invoke(question)

    if not documents:

        return (
            "No relevant information was found "
            "in the Anglican Catechism."
        )

    return "\n\n".join(
        f"[Page {doc.metadata.get('page', 'unknown')}]\n"
        f"{doc.page_content}"
        for doc in documents
    )

# TOOLS
tools = [
    search_bible_verse,
    explain_bible_verse,
    search_catechism
]

# GIVE LLM ACCESS TO TOOLS

llm_with_tools = llm.bind_tools(tools)

# AGENT NODE

def agent(state: AgentState):

    print("Entering agent...")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# AGENT ROUTER

def agent_router(state: AgentState):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):

        return "tools"

    return "end"

# TOPIC ROUTER

def topic_router(state: AgentState):

    if state["on_topic"]:

        return "agent"

    return "off_topic"

# OFF-TOPIC RESPONSE

def off_topic_response(state: AgentState):

    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry, I can only help with "
                    "Anglicanism, the Anglican Catechism, "
                    "Christianity, Christian living, "
                    "Jesus Christ, the Gospel, and the Bible."
                )
            )
        ]
    }

# TOOL NODE

tool_node = ToolNode(tools)


# CHECKPOINTER

checkpointer = MemorySaver()


# BUILD GRAPH
builder = StateGraph(AgentState)


# Add nodes
builder.add_node("classifier",question_classifier)

builder.add_node("agent",agent)

builder.add_node("tools",tool_node)

builder.add_node("off_topic",off_topic_response)

# GRAPH EDGES

# START → CLASSIFIER

builder.add_edge(START,"classifier")


# CLASSIFIER → AGENT / OFF_TOPIC

builder.add_conditional_edges("classifier",topic_router,
    {"agent": "agent","off_topic": "off_topic"})

# AGENT → TOOLS / END

builder.add_conditional_edges("agent",agent_router,
    {"tools": "tools","end": END})

# TOOLS → AGENT

builder.add_edge("tools","agent")


# OFF_TOPIC → END

builder.add_edge("off_topic",END)

# COMPILE GRAPH

graph = builder.compile(checkpointer=checkpointer)


# TEST CHATBOT
if __name__ == "__main__":

    print("\n📖 Anglican Christian AI Assistant")
    print("Ask a question.")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        user_question = input(
            "\nYou: "
        ).strip()

        if user_question.lower() in [
            "exit",
            "quit"
        ]:

            print("Goodbye! 👋")

            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_question
                    )
                ],
                "on_topic": False,
                "context": ""
            },
            config={
                "configurable": {
                    "thread_id": "user-1"
                }
            }
        )

        answer = result["messages"][-1].content

        # Handle structured/list content
        if isinstance(answer, list):

            answer = "".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict)
            )

        print("\nAssistant:")
        print(answer)