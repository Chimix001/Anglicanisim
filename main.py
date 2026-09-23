from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from RagAgent import graph


app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "Christian AI Agent app is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=request.message)
            ],
            "on_topic": False,
            "context": ""
        },
        config={
            "configurable": {
                "thread_id": "api-user"
            }
        }
    )

    return {
        "response": result["messages"][-1].content
    }