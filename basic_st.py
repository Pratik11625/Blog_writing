from __future__ import annotations

import streamlit as st
import os
import operator
from pathlib import Path
from dotenv import load_dotenv
import time

from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
import torch

# 1. Render the sidebar title
st.sidebar.title("📝 Blog Writing Assistant")

# 2. Use the sidebar context for inputs
with st.sidebar:
    # Toggle for Environment Variable vs Manual Input
    use_env_key = st.toggle("Use System API Key")
    
    if use_env_key:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if GROQ_API_KEY:
            st.success("API Key loaded from environment!")
        else:
            st.error("No API Key found in environment variables.")
    else:
        GROQ_API_KEY = st.text_input(
            "Enter Groq API Key", 
            type="password", 
            key="groq_api_key_input"
        )
        if GROQ_API_KEY:
            st.info("Manual API Key active.")

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

# ---------- Models ----------
class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str

# ---------- LLM ----------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=GROQ_API_KEY
)

# ---------- Orchestrator ----------
def orchestrator(state: State):

    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "Create a blog plan with 5 sections. "
                    "Return blog title and tasks."
                    "add a summary brief at the end of the blog."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )

    return {"plan": plan}

# ---------- Fanout ----------
def fanout(state: State):
    return [
        Send(
            "worker",
            {"task": task, "topic": state["topic"], "plan": state["plan"]},
        )
        for task in state["plan"].tasks
    ]

# ---------- Worker ----------
def worker(payload: dict):

    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    section_md = llm.invoke(
        [
            SystemMessage(content="Write a clean Markdown section."),
            HumanMessage(
                content=(
                    f"Blog: {plan.blog_title}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    "Return only markdown section content."
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}

# ---------- Reducer ----------
def reducer(state: State):

    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    filename = title.lower().replace(" ", "_") + ".md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}

# ---------- Graph ----------
g = StateGraph(State)

g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

# ---------- Streamlit UI ----------
st.header("📝 Blog Writing Assistant")

topic = st.text_input(
    "Enter blog topic:",
    placeholder="Example: Future of AI in Healthcare"
)

generate_btn = st.button("🚀 Generate Blog")

if generate_btn and topic:
    with st.spinner("Generating blog..."):
        # result = app.invoke({"topic": topic, "sections": []})

        progress = st.progress(0)
        status = st.empty()

        start_time = time.time()

        status.write("🧠 Creating blog plan...")
        progress.progress(30)

        result = app.invoke({"topic": topic, "sections": []})

        progress.progress(90)
        status.write("✍️ Finalizing blog...")

        progress.progress(100)
        # status.write("✅ Blog ready!")

        elapsed = time.time() - start_time
        st.success(f"⏱️ Generated in {elapsed:.2f} seconds")

        # st.success("📖 Generated Blog")
        st.markdown(result["final"])

        st.download_button(
            "⬇ Download Blog",
            result["final"],
            file_name="blog.md",
            mime="text/markdown",
        )

