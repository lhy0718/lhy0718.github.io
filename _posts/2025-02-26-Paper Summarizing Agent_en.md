---
title: "[Langgraph] This agent summarizes research papers for $0.006."
date: 2025-02-26 22:00:00 +0900
categories:
  - Agents
tags:
  - Langgraph
  - Langchain
  - Agents
  - LLM
---

**Summary:** Let's build an agent using **LangGraph** that automatically summarizes research papers in the desired language.

---

# **Overview**

While experimenting with LangGraph by creating small agents, I developed a **paper summarization agent**. This agent takes a research paper in **PDF format** and summarizes it in a **specified language**.

The overall graph structure is as follows:

```
[PDF Parsing & Section Splitting Node] -> [Multiple Summarization Nodes] -> [Result Merging & Saving Node]
```

The next section explains how the **PDF parsing and section splitting node** processes a PDF.

---

# **Defining the LLM**

This agent will use **GPT-4o-mini**. The LLM is defined as follows:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # Load API keys for OpenAI, LangSmith, etc.

llm = ChatOpenAI(model="gpt-4o-mini")
```

---

# **Parsing the PDF**

LangGraph supports loading and processing **various file formats**, including PDFs ([documentation](https://python.langchain.com/docs/integrations/document_loaders/)).  
For PDFs, **multiple loaders** are available ([list](https://python.langchain.com/docs/integrations/document_loaders/#pdfs)), and I chose **PyPDFLoader** for implementation.

The following code loads the PDF file (the `state` variable will be explained later):

```python
loader = PyPDFLoader(state["file_path"])
pages = []
async for page in loader.alazy_load():
  pages.append(page)
content = "\n".join([page.page_content for page in pages])
```

Since this function is **asynchronous**, the **parent function must also include `async`**, and when executing the graph, we need to use **`ainvoke` instead of `invoke`** (explained later).

---

# **section_spliter (PDF Parsing & Section Splitting Node)**

Before defining the node, we need to define the **State** it will use. The required variables are:

- `file_path`: Path to the PDF file.
- `language`: Target language for summarization.
- `summary`: Stores summarization results.

```python
class OverallState(TypedDict):
    """Overall state class"""

    file_path: str
    language: str
    summary: Annotated[list, operator.add]
```

The `summary` variable is structured for **map-reduce processing**, allowing each summarization node’s output to be merged ([reference](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.ipynb)).

Next, we extract the **table of contents (ToC)** from the document using LLM and split the content into sections accordingly.

```python
res = llm.invoke(
    [
        SystemMessage(
            "Extract the top-level sections of the paper, excluding Acknowledgements and Appendix.\n"
            "Format output as: 1 Introduction\n2 ... "
        ),
        HumanMessage(content),
    ]
)
```

This extracts the **table of contents**. However, we need to:

1. Ensure **"Abstract"** is included in the ToC.
2. Identify the **last section** before "References" to correctly segment the paper.

```python
titles = [line.strip() for line in res.content.splitlines()]
if "Abstract" not in titles:
    titles.insert(0, "Abstract")  # Add Abstract if missing
if "References" not in titles:
    titles.append("References")  # Add References if missing

sections = []
for i, title in enumerate(titles[:-1]):  # Exclude the last section (References)
    section = content.split(title)[1].split(titles[i + 1])[0]  # Extract content between section titles
    sections.append(section)
```

Since the **LLM-generated ToC may not be perfect**, there might be occasional errors, but it works well in most cases.

---

## **Complete Code for section_spliter Node**

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from typing_extensions import Annotated, TypedDict

async def section_spliter(
    state: OverallState,
) -> Command[Literal["section_summarizer"]]:

    loader = PyPDFLoader(state["file_path"])
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    content = "\n".join([page.page_content for page in pages])

    res = llm.invoke(
        [
            SystemMessage(
                "Extract the top-level sections of the paper, excluding Acknowledgements and Appendix.\n"
                "Format output as: 1 Introduction\n2 ... "
            ),
            HumanMessage(content),
        ]
    )

    titles = [line.strip() for line in res.content.splitlines()]
    if "Abstract" not in titles:
        titles.insert(0, "Abstract")
    if "References" not in titles:
        titles.append("References")

    sections = []
    for i, title in enumerate(titles[:-1]):
        section = content.split(title)[1].split(titles[i + 1])[0]
        sections.append(section)

    return Command(
        goto=[
            Send(
                "section_summarizer",
                {"title": t, "section": s, "language": state["language"]},
            )
            for t, s in zip(titles, sections)
        ]
    )
```

---

## **Key Points in the Code**

- The function returns a **`Command[Literal["section_summarizer"]]`**, explicitly specifying its return type.
- Since `Command` is returned, **graph edges are not manually added to the graph builder**.
- The `goto` statement **routes** each section’s title, content, and target language to the **"section_summarizer" node**.

The next step is to define the **section summarization node** and handle merging the results.

## **section_summarizer (Summarization Node)**

The **section_summarizer** node uses **LLM** to summarize each section it receives.

- If the title is `"Abstract"`, the summary is **1-2 sentences** long.
- Otherwise, the section is summarized **without length restrictions** in **Markdown format**.

### **Complete Code for section_summarizer**

```python
class SectionState(TypedDict):
    """Section state class"""

    title: str
    section: str
    language: str

def section_summarizer(state: SectionState):

    prompt = ""
    if state["title"] == "Abstract":
        prompt = f"Summarize the abstract in {state['language']} in 1-2 sentences."
    else:
        prompt = f"Summarize Section '{state['title']}' in {state['language']} in .md format (title should start with '# {state['title']}'). Wrap equations in $$."

    res = llm.invoke(
        [
            SystemMessage(prompt),
            HumanMessage(state["section"]),
        ]
    )
    return {"summary": [res.content]}
```

---

## **gatherer (Result Merging & Saving Node)**

The **gatherer** node processes and merges all collected summaries stored in `state["summary"]`, then saves the final output as a **Markdown file**.

### **Complete Code for gatherer**

````python
import os

def gatherer(state: OverallState):
    summaries = []
    for section in state["summary"]:
        section = section.strip()
        if section.startswith("```markdown"):
            section = section.split("```markdown")[1]
        if section.startswith("```md"):
            section = section.split("```md")[1]
        if section.startswith("```"):
            section = section.split("```")[1]
        if section.endswith("```"):
            section = section.rsplit("```")[0]
        summaries.append(section)

    total_summary = "---\n\n" + "\n\n---\n\n".join(summaries)

    with open(f"{os.path.splitext(state['file_path'])[0]}.md", "w") as f:
        f.write(total_summary)
````

### **Post-Processing Steps**

1. **Removes unnecessary Markdown code block formatting** (` ```markdown` and ` ``` `).
2. **Adds section separators (`---`)** for readability.
3. **Saves the output as an `.md` file** with the same name as the original PDF.

---

## **Building and Executing the Graph**

Now, let's assemble all the nodes into a **LangGraph pipeline**.

### **Graph Building Code**

```python
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(OverallState)

graph_builder.add_node("section_spliter", section_spliter)
graph_builder.add_edge(START, "section_spliter")

graph_builder.add_node("section_summarizer", section_summarizer)
graph_builder.add_node("gatherer", gatherer)

graph_builder.add_edge("section_summarizer", "gatherer")
graph_builder.add_edge("gatherer", END)

graph = graph_builder.compile()
```

---

### **Visualizing the Graph**

```python
from IPython.display import Image, display

display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

This visualization shows how the **PDF is processed into structured summaries**.

![graph-output](https://github.com/user-attachments/assets/17b38111-7202-49f0-a26b-86202eadfefb)

---

### **Executing the Graph**

Since the first node is **asynchronous**, we must use `ainvoke` when running the graph:

```python
await graph.ainvoke({"file_path": "VaRMI.pdf", "language": "Korean"})
```

A **Markdown file** with the summarized paper should now be created! 🎉

---

## **Handling Errors**

If an **error occurs**, it is likely due to **incorrect ToC extraction by the LLM**. Simply **rerun the process**, and it should work fine.

## **Conclusion**

This project demonstrates how to **build an automated research paper summarization agent using LangGraph**.

- The agent efficiently **parses PDFs**, **extracts sections**, **summarizes them using GPT-4o-mini**, and **merges the results into a Markdown file**.
- In testing, **summarizing a paper costs approximately $0.006**, making it a **cost-effective and fast solution** for quick paper review.
- This approach is especially useful when **reading numerous papers quickly and extracting key insights in a preferred language**.

---

## **Full Code**

Here’s the **complete implementation** of the LangGraph-based research paper summarization agent:

````python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

import operator
import os
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from typing_extensions import Annotated, TypedDict


class OverallState(TypedDict):
    """Overall state class"""

    file_path: str
    language: str
    summary: Annotated[list, operator.add]


class SectionState(TypedDict):
    """Section state class"""

    title: str
    section: str
    language: str


async def section_spliter(
    state: OverallState,
) -> Command[Literal["section_summarizer"]]:

    loader = PyPDFLoader(state["file_path"])
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    content = "\n".join([page.page_content for page in pages])

    res = llm.invoke(
        [
            SystemMessage(
                "Extract the top-level sections of the paper, excluding Acknowledgements and Appendix.\n"
                "Format output as: 1 Introduction\n2 ..."
            ),
            HumanMessage(content),
        ]
    )

    titles = [line.strip() for line in res.content.splitlines()]
    if "Abstract" not in titles:
        titles.insert(0, "Abstract")
    if "References" not in titles:
        titles.append("References")

    sections = []
    for i, title in enumerate(titles[:-1]):
        section = content.split(title)[1].split(titles[i + 1])[0]
        sections.append(section)

    return Command(
        goto=[
            Send(
                "section_summarizer",
                {"title": t, "section": s, "language": state["language"]},
            )
            for t, s in zip(titles, sections)
        ]
    )


def section_summarizer(state: SectionState):

    prompt = ""
    if state["title"] == "Abstract":
        prompt = f"Summarize the abstract in {state['language']} in 1-2 sentences."
    else:
        prompt = f"Summarize Section '{state['title']}' in {state['language']} in .md format (title should start with '# {state['title']}'). Wrap equations in $$."

    res = llm.invoke(
        [
            SystemMessage(prompt),
            HumanMessage(state["section"]),
        ]
    )
    return {"summary": [res.content]}


def gatherer(state: OverallState):
    summaries = []
    for section in state["summary"]:
        section = section.strip()
        if section.startswith("```markdown"):
            section = section.split("```markdown")[1]
        if section.startswith("```md"):
            section = section.split("```md")[1]
        if section.startswith("```"):
            section = section.split("```")[1]
        if section.endswith("```"):
            section = section.rsplit("```")[0]
        summaries.append(section)

    total_summary = "---\n\n" + "\n\n---\n\n".join(summaries)
    with open(f"{os.path.splitext(state['file_path'])[0]}.md", "w") as f:
        f.write(total_summary)


graph_builder = StateGraph(OverallState)
graph_builder.add_node("section_spliter", section_spliter)
graph_builder.add_edge(START, "section_spliter")
graph_builder.add_node("section_summarizer", section_summarizer)
graph_builder.add_node("gatherer", gatherer)
graph_builder.add_edge("section_summarizer", "gatherer")
graph_builder.add_edge("gatherer", END)
graph = graph_builder.compile()

from IPython.display import Image, display

display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

await graph.ainvoke({"file_path": "VaRMI.pdf", "language": "Korean"})
````

---

## **Next Steps**

- **Enhance summarization quality** by fine-tuning prompts or using **different LLMs** for improved summarization accuracy.
- **Optimize execution speed and cost**, possibly by using smaller models for initial processing before invoking a more powerful model.
- **Expand functionality** to include **PDF metadata extraction, citation analysis, or automated tagging** for research papers.

This project demonstrates **LangGraph's power in building modular, efficient, and scalable AI workflows** for automating academic paper summarization. 🚀
