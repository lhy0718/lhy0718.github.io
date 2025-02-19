---
title: "[Langgraph] Creating an Automatic Translation Agent for GitHub Code Documentation"
date: 2025-02-19 00:00:00 +0900
categories:
  - Agents
tags:
  - Langgraph
  - Agents
  - LLM
---

summary: Sharing the experience of using Langgraph's hierarchical model to build an automatic code documentation translation agent.

## Overview
I am currently working on building an agent using Langgraph to automatically translate documentation files in code repositories, such as `.md` and `.ipynb`. Specifically, this agent extracts a list of documents to be translated and utilizes an LLM in parallel to translate each document. However, I encountered the following issues when attempting to perform translations using a single Langgraph agent:

- A single graph shares one state, causing the translation results of different documents to be mixed in the same state.
- Consequently, parallel processing of document translation was not possible.

To address this problem, I experimented with various approaches and found that a hierarchical model composed of subgraphs effectively resolved the issue.

## Hierarchical Model

According to Langgraph's [official documentation](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#multi-agent-architectures), a hierarchical architecture is defined as a "supervisor of supervisors" in a multi-agent system.
![image](https://github.com/user-attachments/assets/e6e638ac-f542-4299-ae89-15b1637d96bb)

The agent I am building consists of a supervisor that distributes documents for translation and sub-agents that handle the translation of each document. Additionally, since the `gpt-4o-mini` model struggles with long documents, a supervisor is required to segment the documents into appropriate lengths before passing them to the translation LLM. Each sub-agent must be dynamically assigned using the [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) function, and each translation LLM must also be dynamically allocated using Send.

The final graph structure of my agent is as follows. (For some reason, the `cell_translator` and `text_translator` nodes are not rendering their respective `cell_synthesizer` and `text_synthesizer` nodes in the image. I am unsure if this is a Langgraph bug.)
![agent](https://github.com/user-attachments/assets/db29b773-71a2-4a16-a004-c09d46d96154)

## Implementation

Now, I will explain the code that creates each node.

### Orchestrator

The Orchestrator collects the list of documents to be translated and sends them to subgraphs for processing.

The required libraries for implementation are as follows:

```python
import json
import operator
import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.types import Command, Send
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict
```

The code for the Orchestrator is as follows:

```python
class OverallState(TypedDict):
    """Overall state class"""

    root_dir: str  # Root directory
    excludes: list[str]  # Directories to exclude
    target_files_ext: list[str]  # Target file extensions for translation

def orchestrator(
    state: OverallState,
) -> Command[Literal["text_file_translator", "ipynb_file_translator"]]:
    """Node managing the overall process"""

    global progress_bar

    src_paths = []
    dst_paths = []
    for dir, sub_dir, files in os.walk(state["root_dir"]):
        if os.path.abspath(dir) in [os.path.abspath(e) for e in state["excludes"]]:
            continue
        for file in files:
            if os.path.join(os.path.abspath(dir), file) in [
                os.path.abspath(e) for e in state["excludes"]
            ]:
                continue
            src_path = os.path.join(dir, file)
            dst_path = set_save_path(dir, file)
            if (
                os.path.splitext(file)[1] in state["target_files_ext"]
                and os.path.isfile(src_path)
                and not is_translated_file(src_path)
            ):
                src_paths.append(src_path)
                dst_paths.append(dst_path)

    progress_bar = tqdm(
        total=len(src_paths), desc="Translation Progress", position=0, leave=True
    )

    return Command(
        goto=[
            (
                Send("ipynb_file_translator", {"src_path": s, "dst_path": d})
                if os.path.splitext(s)[1] == ".ipynb"
                else Send("text_file_translator", {"src_path": s, "dst_path": d})
            )
            for s, d in zip(src_paths, dst_paths)
        ],
    )
```
When analyzing the code, we see that the function returns a `Command` object and explicitly defines its output with `-> Command[Literal["text_file_translator", "ipynb_file_translator"]]`. This allows the graph builder to automatically create edges when calling `.add_node()`, eliminating the need for `.add_edge()`. **This is only possible when the node returns a `Command` function.**

The return statement in the code returns a `Command` object, defining conditional edges via the `goto` argument. The `goto` argument returns a list of `Send` objects, enabling dynamic node creation. The `Send` function contains `ipynb_file_translator` and `text_file_translator`, which are the subgraphs explained later.

At the bottom of the full implementation, the graph builder is structured as follows:

```python
    graph_builder = StateGraph(OverallState)

    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_edge(START, "orchestrator")

    ... # Subgraph definitions

    graph_builder.add_node("text_file_translator", subgraph1)
    graph_builder.add_node("ipynb_file_translator", subgraph2)
    graph = graph_builder.compile()
```

Next, let's go over the implementation of each subgraph.

### `text_assigner`

The `text_assigner` function segments the document into appropriately sized chunks and forwards them to `text_translator`, the node containing the LLM. Here is the implementation of `text_assigner`:

```python
class AssignerState(TypedDict):
    """Assignment state class"""

    completed_translation: Annotated[list, operator.add]
    src_path: str
    dst_path: str

class TextAssignerState(AssignerState):
    whitespace_prefix_list: list[str]
    whitespace_suffix_list: list[str]
    chunks: list[str]

def text_assigner(state: TextAssignerState) -> Command[Literal["text_translator"]]:
    """Splits text into chunks and assigns them to the Translator"""

    with open(state["src_path"], "r") as f:
        text = f.read()

    chunks = []
    whitespace_prefix_list = []
    whitespace_suffix_list = []

    while text:
        lines = text.split("\n")
        chunk = "\n".join(lines[:100])
        if (
            chunk.strip().startswith("```") and chunk.count("```") == 1
        ):  # If a code block starts but does not end within 100 lines -> translate the whole code block
            chunk = "```".join(text.split("```")[:2]) + "```"
        elif (
            chunk.count("```") % 2 == 1
        ):  # If a code block is not closed -> translate only up to the code start
            chunk = "```".join(chunk.split("```")[:-1])
        if match := re.search(IMG_REGEX, chunk):
            if match.start() == 0:  # If an image appears at the beginning
                chunk = chunk[: match.end()]
            else:  # If an image appears in the middle
                chunk = chunk[: match.start()]
        chunks.append(chunk)
        whitespace_prefix_list.append(get_l_whitespace(chunk))
        whitespace_suffix_list.append(get_r_whitespace(chunk))
        text = text[len(chunk):]

    return Command(
        update={
            "chunks": chunks,
            "whitespace_prefix_list": whitespace_prefix_list,
            "whitespace_suffix_list": whitespace_suffix_list,
        },
        goto=[
            Send(
                "text_translator",
                {
                    "text": c.strip(),
                },
            )
            for c in chunks
        ],
    )
```

While the internal logic is complex due to handling code blocks in the text, the key takeaway is that, similar to the `Orchestrator`, it returns a `Command` object and dynamically creates nodes through the `goto` argument. The `Send` function is used to dynamically create `text_translator` nodes and pass the `text` to them.

### cell_assigner

```python
class CellAssignerState(AssignerState):
    cells: list[dict]

def cell_assigner(state: CellAssignerState) -> Command[Literal["cell_translator"]]:
    """Function that divides an ipynb file into cells and assigns them to the Translator"""

    with open(state["src_path"], "r") as f:
        data = json.load(f)

    cells = data["cells"]

    return Command(
        update={
            "cells": cells,
        },
        goto=[
            Send(
                "cell_translator",
                {
                    "text": "".join(c["source"]),
                    "type": c["cell_type"],
                },
            )
            for c in cells
        ],
    )
```

The structure of `cell_assigner` is almost identical to `text_assigner`. However, `cell_assigner` divides the ipynb file into cells and passes the text and cell_type within each cell to `cell_translator`.

### Translators

The translation nodes use LLM to perform text translation. Since the structures of `text_translator` and `cell_translator` are similar, the redundant parts have been extracted into a `translator` function.

```python
class TranslatorState(TypedDict):
    """Translation state class"""

    text: str
    type: Literal["markdown", "code"]

def translator(state: TranslatorState, goto: str) -> Command:
    """Function to translate a piece of text"""

    text = state["text"]
    if text.strip() == "":
        return Command(update={"completed_translation": [""]}, goto=goto)

    if re.search(IMG_REGEX, text):
        return Command(
            update={"completed_translation": [text]},
            goto=goto,
        )

    translation = llm.invoke(
        [
            SystemMessage(content=TRANSLATE_PROMPT),
            HumanMessage(content=text),
        ]
    )

    return Command(
        update={"completed_translation": [translation.content]},
        goto=goto,
    )


def text_translator(state: TranslatorState) -> Command[Literal["text_synthesizer"]]:
    return translator(state, "text_synthesizer")


def cell_translator(state: TranslatorState) -> Command[Literal["cell_synthesizer"]]:
    if state["type"] == "code":
        return Command(
            update={"completed_translation": [state["text"]]},
            goto="cell_synthesizer",
        )
    return translator(state, "cell_synthesizer")
```

Both translators store the translated results in the "completed_translation" field and pass them as a Command object to the respective synthesizer.

Next, let's look at how the translated results are merged.

### text_synthesizer

```python
def text_synthesizer(state: TextAssignerState):
    """Function to synthesize the translated document"""

    global prograss_bar

    completed_translation = [
        r + line + l
        for r, line, l in zip(
            state["whitespace_prefix_list"],
            state["completed_translation"],
            state["whitespace_suffix_list"],
        )
    ]
    final_translation = KOREAN_TRANSLATED_MSG + "".join(completed_translation)

    with open(state["dst_path"], "w") as f:
        f.write(final_translation)

    prograss_bar.update(1)
    return None
```

This code also involves a complex logic to combine text, but the important point is that it uses `completed_translation` from `TextAssignerState` to create the final text. This is because `TextAssignerState` is defined as follows:

```python
class AssignerState(TypedDict):
    """Assignment state class"""

    completed_translation: Annotated[list, operator.add]
    src_path: str
    dst_path: str


class TextAssignerState(AssignerState):
    whitespace_prefix_list: list[str]
    whitespace_suffix_list: list[str]
    chunks: list[str]
```

The `completed_translation` is defined as `Annotated[list, operator.add]`. This means that the values returned by the dynamically created translators are successively added to a single list.

A unique aspect of `text_synthesizer` is the inclusion of `KOREAN_TRANSLATED_MSG` at the beginning of the text. This prevents the agent from translating the text again in the next run.

### cell_synthesizer

The `cell_synthesizer` is written in a similar manner:

```python
class CellAssignerState(AssignerState):
    cells: list[dict]

def cell_synthesizer(state: CellAssignerState):
    """Function to synthesize the translated ipynb file"""

    global prograss_bar

    data = {"cells": []}
    for source, cell in zip(state["completed_translation"], state["cells"]):
        if cell["cell_type"] == "code":
            source = re.sub("^```.*", "", source, flags=re.M).strip()
        source = [line + "\n" for line in source.split("\n")]
        cell["source"] = source
        data["cells"].append(cell)

    data["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [KOREAN_TRANSLATED_MSG],
        },
    )

    with open(state["dst_path"], "w") as f:
        json.dump(data, f)

    prograss_bar.update(1)
    return None
```

In `cell_synthesizer`, the `KOREAN_TRANSLATED_MSG` is added as the first cell, indicating that the document has been translated.

### Compiling the Entire Graph

The code for compiling the entire graph is as follows:

```python
    graph_builder = StateGraph(OverallState)

    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_edge(START, "orchestrator")

    subgraph1_builder = StateGraph(TextAssignerState)
    subgraph1_builder.add_node("text_assigner", text_assigner)
    subgraph1_builder.add_edge(START, "text_assigner")
    subgraph1_builder.add_node("text_translator", text_translator)
    subgraph1_builder.add_node("text_synthesizer", text_synthesizer)
    subgraph1 = subgraph1_builder.compile()

    subgraph2_builder = StateGraph(CellAssignerState)
    subgraph2_builder.add_node("cell_assigner", cell_assigner)
    subgraph2_builder.add_edge(START, "cell_assigner")
    subgraph2_builder.add_node("cell_translator", cell_translator)
    subgraph2_builder.add_node("cell_synthesizer", cell_synthesizer)
    subgraph2 = subgraph2_builder.compile()

    graph_builder.add_node("text_file_translator", subgraph1)
    graph_builder.add_node("ipynb_file_translator", subgraph2)
    graph = graph_builder.compile()
```

### Conclusion

By constructing this hierarchical model, an agent was created to translate documents in parallel using Langgraph. This allowed for utilizing the powerful features of Langgraph while solving complex problems. Now, this agent can be used to translate documents from code repositories.

### Full Code


```python
# Initialize LLM
import json
import operator
import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.types import Command, Send
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

#### Set the root directory of the documents to be translated and the list of directories to exclude ####

root_dir = "../langgraph/"
excludes = [
    "../langgraph/docs/docs/reference",
    "../langgraph/docs/docs/index.md",
]

#####################################################################

GRAPH_OUTPUT_PATH = "agent.png"
LLM = "gpt-4o-mini"
KOREAN_TRANSLATED_MSG = "_Translated to Korean by machine_\n\n"
TRANSLATE_PROMPT = "Only 'translate' the provided text into Korean."

load_dotenv()  # load environment variables from `.env` file

llm = ChatOpenAI(model=LLM)

prograss_bar = None


class OverallState(TypedDict):
    """Overall state class"""

    root_dir: str  # Root directory
    excludes: list[str]  # Directories to exclude
    target_files_ext: list[str]  # Extensions of files to be translated


class AssignerState(TypedDict):
    """Assignment state class"""

    completed_translation: Annotated[list, operator.add]
    src_path: str
    dst_path: str


class TextAssignerState(AssignerState):
    whitespace_prefix_list: list[str]
    whitespace_suffix_list: list[str]
    chunks: list[str]


class CellAssignerState(AssignerState):
    cells: list[dict]


class TranslatorState(TypedDict):
    """Translation state class"""

    text: str
    type: Literal["markdown", "code"]


def set_save_path(dir: str, file: str) -> str:
    """Function to set the save path"""
    return os.path.join(dir, file)


def is_translated_file(src_path: str) -> bool:
    """Function to check if the file has already been translated"""

    if os.path.splitext(src_path)[1] == ".ipynb":
        with open(src_path, "r") as f:
            data = json.load(f)
        if data["cells"][0]["source"][0].strip() == KOREAN_TRANSLATED_MSG.strip():
            return True

    with open(src_path, "r") as f:
        text = f.read()
    if text.startswith(KOREAN_TRANSLATED_MSG):
        return True

    return False


def orchestrator(
    state: OverallState,
) -> Command[Literal["text_file_translator", "ipynb_file_translator"]]:
    """Node that manages the entire process"""

    global prograss_bar

    src_paths = []
    dst_paths = []
    for dir, sub_dir, files in os.walk(state["root_dir"]):
        if os.path.abspath(dir) in [os.path.abspath(e) for e in state["excludes"]]:
            continue
        for file in files:
            if os.path.join(os.path.abspath(dir), file) in [
                os.path.abspath(e) for e in state["excludes"]
            ]:
                continue
            src_path = os.path.join(dir, file)
            dst_path = set_save_path(dir, file)
            if (
                os.path.splitext(file)[1] in state["target_files_ext"]
                and os.path.isfile(src_path)
                and not is_translated_file(src_path)
            ):
                src_paths.append(src_path)
                dst_paths.append(dst_path)

    prograss_bar = tqdm(
        total=len(src_paths), desc="Translation progress", position=0, leave=True
    )

    return Command(
        goto=[
            (
                Send("ipynb_file_translator", {"src_path": s, "dst_path": d})
                if os.path.splitext(s)[1] == ".ipynb"
                else Send("text_file_translator", {"src_path": s, "dst_path": d})
            )
            for s, d in zip(src_paths, dst_paths)
        ],
    )


def get_l_whitespace(text: str) -> str:
    """Function to return the left whitespace of a string"""

    return text[: len(text) - len(text.lstrip())]


def get_r_whitespace(text: str) -> str:
    """Function to return the right whitespace of a string"""

    return text[len(text.rstrip()) :]


IMG_REGEX = r"!\[.*\]\(data:image.*\)"


def text_assigner(state: TextAssignerState) -> Command[Literal["text_translator"]]:
    """Function to chunk the text and assign it to the Translator"""

    with open(state["src_path"], "r") as f:
        text = f.read()

    chunks = []
    whitespace_prefix_list = []
    whitespace_suffix_list = []

    while text:
        lines = text.split("\n")
        chunk = "\n".join(lines[:100])
        if (
            chunk.strip().startswith("```") and chunk.count("```") == 1
        ):  # If a code block starts but doesn't end within 100 lines -> translate the whole code
            chunk = "```".join(text.split("```")[:2]) + "```"
        elif (
            chunk.count("```") % 2 == 1
        ):  # If a code block doesn't end -> translate up to the start of the code block
            chunk = "```".join(chunk.split("```")[:-1])
        if match := re.search(IMG_REGEX, chunk):
            if match.start() == 0:  # If an image is at the first line
                chunk = chunk[: match.end()]
            else:  # If an image is in the middle of the chunk
                chunk = chunk[: match.start()]
        chunks.append(chunk)
        whitespace_prefix_list.append(get_l_whitespace(chunk))
        whitespace_suffix_list.append(get_r_whitespace(chunk))
        text = text[len(chunk) :]

    return Command(
        update={
            "chunks": chunks,
            "whitespace_prefix_list": whitespace_prefix_list,
            "whitespace_suffix_list": whitespace_suffix_list,
        },
        goto=[
            Send(
                "text_translator",
                {
                    "text": c.strip(),
                },
            )
            for c in chunks
        ],
    )


def cell_assigner(state: CellAssignerState) -> Command[Literal["cell_translator"]]:
    """Function to divide ipynb files into cells and assign them to the Translator"""

    with open(state["src_path"], "r") as f:
        data = json.load(f)

    cells = data["cells"]

    return Command(
        update={
            "cells": cells,
        },
        goto=[
            Send(
                "cell_translator",
                {
                    "text": "".join(c["source"]),
                    "type": c["cell_type"],
                },
            )
            for c in cells
        ],
    )


def translator(state: TranslatorState, goto: str) -> Command:
    """Function to translate text chunks"""

    text = state["text"]
    if text.strip() == "":
        return Command(update={"completed_translation": [""]}, goto=goto)

    if re.search(IMG_REGEX, text):
        return Command(
            update={"completed_translation": [text]},
            goto=goto,
        )

    translation = llm.invoke(
        [
            SystemMessage(content=TRANSLATE_PROMPT),
            HumanMessage(content=text),
        ]
    )

    return Command(
        update={"completed_translation": [translation.content]},
        goto=goto,
    )


def text_translator(state: TranslatorState) -> Command[Literal["text_synthesizer"]]:
    return translator(state, "text_synthesizer")


def cell_translator(state: TranslatorState) -> Command[Literal["cell_synthesizer"]]:
    if state["type"] == "code":
        return Command(
            update={"completed_translation": [state["text"]]},
            goto="cell_synthesizer",
        )
    return translator(state, "cell_synthesizer")


def text_synthesizer(state: TextAssignerState):
    """Function to synthesize the translated document"""

    global prograss_bar

    completed_translation = [
        r + line + l
        for r, line, l in zip(
            state["whitespace_prefix_list"],
            state["completed_translation"],
            state["whitespace_suffix_list"],
        )
    ]
    final_translation = KOREAN_TRANSLATED_MSG + "".join(completed_translation)

    with open(state["dst_path"], "w") as f:
        f.write(final_translation)

    prograss_bar.update(1)
    return None


def cell_synthesizer(state: CellAssignerState):
    """Function to synthesize the translated ipynb file"""

    global prograss_bar

    data = {"cells": []}
    for source, cell in zip(state["completed_translation"], state["cells"]):
        if cell["cell_type"] == "code":
            source = re.sub("^```.*", "", source, flags=re.M).strip()
        source = [line + "\n" for line in source.split("\n")]
        cell["source"] = source
        data["cells"].append(cell)

    data["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [KOREAN_TRANSLATED_MSG],
        },
    )

    with open(state["dst_path"], "w") as f:
        json.dump(data, f)

    prograss_bar.update(1)
    return None


if __name__ == "__main__":
    graph_builder = StateGraph(OverallState)

    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_edge(START, "orchestrator")

    subgraph1_builder = StateGraph(TextAssignerState)
    subgraph1_builder.add_node("text_assigner", text_assigner)
    subgraph1_builder.add_edge(START, "text_assigner")
    subgraph1_builder.add_node("text_translator", text_translator)
    subgraph1_builder.add_node("text_synthesizer", text_synthesizer)
    subgraph1 = subgraph1_builder.compile()

    subgraph2_builder = StateGraph(CellAssignerState)
    subgraph2_builder.add_node("cell_assigner", cell_assigner)
    subgraph2_builder.add_edge(START, "cell_assigner")
    subgraph2_builder.add_node("cell_translator", cell_translator)
    subgraph2_builder.add_node("cell_synthesizer", cell_synthesizer)
    subgraph2 = subgraph2_builder.compile()

    graph_builder.add_node("text_file_translator", subgraph1)
    graph_builder.add_node("ipynb_file_translator", subgraph2)
    graph = graph_builder.compile()

    graph.get_graph(xray=True).draw_mermaid_png(output_file_path=GRAPH_OUTPUT_PATH)

    # Run the graph
    graph.invoke(
        {
            "root_dir": root_dir,
            "excludes": excludes,
            "target_files_ext": [".md", ".ipynb"],
        }
    )
    prograss_bar.close()
```
