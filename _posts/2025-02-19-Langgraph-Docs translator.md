---
title: "[Langgraph] Github 코드 문서 자동 번역 에이전트 제작"
date: 2025-02-19 00:00:00 +0900
categories:
  - Agents
tags:
  - Langgraph
  - Agents
  - LLM
---

요약: 코드 문서 자동 번역 에이전트를 만들기 위해 Langgraph의 계층적 모델을 사용한 경험을 공유한다.

## 개요
나는 현재 Langgraph를 사용하여 코드 저장소에 있는 .md, .ipynb 등의 형식을 가진 문서들을 자동으로 번역하는 에이전트를 만드려고 한다. 상세하게는 번역 대상이 되는 문서들의 리스트를 뽑고, 병렬적으로 LLM을 사용하여 각 문서를 번역하는 역할을 하는 에이전트이다. 그런데 단일 그래프로 구성된 Langgraph 에이전트를 통해 번역을 수행하려고 했더니 다음과 같은 문제점들이 있었다.

- 단일 그래프는 하나의 State를 공유하므로, 각 문서에 대한 번역 결과가 한 State에 저장되어 번역 결과가 뒤섞이는 문제가 발생한다.
- 즉, 문서 번역에 대한 병렬적 처리가 불가능했다.

그래서 이 문제를 해결하기 위해 정말 다양한 시도들을 해보았고, subgraph들로 구성된 계층적 구조의 모델이 문제를 해결한다는 것을 알게 되었다.

## Hierarchical model

일단 Langgraph의 [공식문서](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#multi-agent-architectures) 에서는 "a supervisor of supervisors"를 사용하여 multi-agent system을 정의한 것을 Hierarchical Architecture라고 표현하고 있다.
![image](https://github.com/user-attachments/assets/e6e638ac-f542-4299-ae89-15b1637d96bb)

내가 만들어야하는 에이전트는 번역할 문서를 분배하는 superviser와 각 문서를 번역하는 sub-agent로 구성되어야 한다. 또한 내가 사용하는 gpt-4o-mini모델은 긴 문서는 번역하기 어려워하므로, 문서를 적절한 길이로 잘라서 번역 LLM에 입력해주는 supervisor가 필요하다. 각 sub-agent는 [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) 함수를 통해 동적으로 할당되어야 하고, 각 번역 LLM 또한 Send를 사용하여 동적으로 할당되어야 한다.

내가 만든 에이전트의 최종적인 그래프 모습은 다음과 같다. (왜인지는 모르겠지만, cell_translator와 text_translator 각각의 아래 연결된 cell_synthesizer, text_synthesizer node가 이미지에 랜더링 되지 않았다. Langgraph의 오류인가 모르겠다.)
![agent](https://github.com/user-attachments/assets/db29b773-71a2-4a16-a004-c09d46d96154)

## 구현

이제 각 노드를 생성하는 코드를 설명하겠다.

### Orchestrator

Orchestrator는 번역할 문서의 리스트를 수집하고, 각 문서를 subgraph에 보내 번역할 수 있도록 하는 역할을 한다.

먼저 구현에 필요한 라이브러리들은 모두 아래와 같다.

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

그리고 Orchestrator의 코드는 다음과 같다.

```python

class OverallState(TypedDict):
    """전체 상태 클래스"""

    root_dir: str  # 루트 디렉터리
    excludes: list[str]  # 제외할 디렉터리
    target_files_ext: list[str]  # 번역 대상 파일 확장자

def orchestrator(
    state: OverallState,
) -> Command[Literal["text_file_translator", "ipynb_file_translator"]]:
    """전체 프로세스를 관리하는 노드"""

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
        total=len(src_paths), desc="번역 진행도", position=0, leave=True
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

코드를 살펴보면 먼저 함수가 Command 객체를 리턴하고, `-> Command[Literal["text_file_translator", "ipynb_file_translator"]]:`으로 output을 명시하고 있음을 볼 수 있다. 이를 통해 그래프 빌더에 .add_node()만으로도 노드가 엣지를 자동 생성할 수 있게 된다. 즉 .add_edge가 필요 없게 된다. **이는 노드가 Command 함수를 리턴할때만 가능하다.**

코드의 return 부분은 Command 객체를 리턴하고 있으며, goto argument를 통해 조건부 엣지를 정의하고 있다. 또한 goto argument가 Send 객체의 list를 리턴하고 있는데, 이러면 동적으로 노드를 생성할 수 있다. 또한 Send함수 안에 들어가는 `ipynb_file_translator`와 `text_file_translator` 는 아래서 설명하게될 subgraph 들이다.

전체 코드의 맨 아래에 그래프 빌더는 다음과 같이 구성한다.

```python
    graph_builder = StateGraph(OverallState)

    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_edge(START, "orchestrator")

    ... # subgraph 정의되어 있는 부분

    graph_builder.add_node("text_file_translator", subgraph1)
    graph_builder.add_node("ipynb_file_translator", subgraph2)
    graph = graph_builder.compile()
```

다음은 각 subgraph들의 구현 설명이다.

### text_assigner

`text_assigner`는 번역할 문서를 적절한 길이로 잘라주고 LLM을 담고있는 노드인 `text_translator`에 전달한다. `text_assigner`의 코드는 다음과 같다.

````python
class AssignerState(TypedDict):
    """할당 상태 클래스"""

    completed_translation: Annotated[list, operator.add]
    src_path: str
    dst_path: str

class TextAssignerState(AssignerState):
    whitespace_prefix_list: list[str]
    whitespace_suffix_list: list[str]
    chunks: list[str]

def text_assigner(state: TextAssignerState) -> Command[Literal["text_translator"]]:
    """텍스트를 청킹하여 Translator에 할당하는 함수"""

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
        ):  # 코드 블록이 시작되지만 100줄 안에 끝나지 않은 경우 -> 코드를 통째로 번역
            chunk = "```".join(text.split("```")[:2]) + "```"
        elif (
            chunk.count("```") % 2 == 1
        ):  # 코드 블록이 끝나지 않은 경우 -> 코드 시작 전까지 번역
            chunk = "```".join(chunk.split("```")[:-1])
        if match := re.search(IMG_REGEX, chunk):
            if match.start() == 0:  # 이미지가 첫 줄에 있는 경우
                chunk = chunk[: match.end()]
            else:  # 이미지가 중간에 있는 경우
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
````

텍스트 안의 코드블록 때문에 내부 로직이 복잡하긴 하지만, 여기서 중요한 것은 Ocherstrator와 마찬가지로 Command 객체를 리턴하고 있으며, goto argument를 통해 동적으로 노드를 생성하고 있다는 것이다. 또한 Send 함수를 통해 동적으로 생성되는 `text_translator`에 `text`를 전달하고 있다.

### cell_assigner

```python
class CellAssignerState(AssignerState):
    cells: list[dict]

def cell_assigner(state: CellAssignerState) -> Command[Literal["cell_translator"]]:
    """ipynb 파일을 셀 단위로 나누어 Translator에 할당하는 함수"""

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

`cell_assigner`의 구조는 `text_assigner`와 거의 동일하다. 다만 `cell_assigner`는 ipynb 파일을 셀 단위로 나누고, 셀 안의 text와 cell_type을 `cell_translator`에 전달한다.

### Translators

번역 노드는 LLM을 사용하여 텍스트 번역을 수행한다. `text_translator`와 `cell_translator`의 구조가 비슷하여 중복되는 부분을 `translator` 함수로 빼주었다.

```python
class TranslatorState(TypedDict):
    """번역 상태 클래스"""

    text: str
    type: Literal["markdown", "code"]

def translator(state: TranslatorState, goto: str) -> Command:
    """텍스트 조각을 번역하는 함수"""

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

두 translator 모두 각각의 synthesizer로 번역된 결과를 "completed_translation"에 담아 Command 객체로 전달한다.

번역된 결과가 어떻게 합쳐지는지는 다음을 보자.

### text_synthesizer

```python
def text_synthesizer(state: TextAssignerState):
    """번역된 문서를 합성하는 함수"""

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

이 코드 또한 텍스트를 합치는 로직이 복잡하긴 하지만, 중요한 것은 `TextAssignerState` 안에 있는 `completed_translation`를 전체 텍스트로 활용하고 있다는 것이다. 이는 `TextAssignerState`가 다음과 같이 정의되었기 때문이다.

```python
class AssignerState(TypedDict):
    """할당 상태 클래스"""

    completed_translation: Annotated[list, operator.add]
    src_path: str
    dst_path: str


class TextAssignerState(AssignerState):
    whitespace_prefix_list: list[str]
    whitespace_suffix_list: list[str]
    chunks: list[str]
```

`completed_translation`는 `Annotated[list, operator.add]`으로 정의되어있다. 이는 동적으로 생성되었던 translator들이 리턴된 값들이 차례대로 하나의 list에 합쳐지는 것을 의미한다.

`text_synthesizer`에서 특이한 점은 `KOREAN_TRANSLATED_MSG`을 텍스트의 맨 앞에 넣는것인데, 이는 번역된 텍스트를 다음 실행 떄 에이전트가 다시 번역하는 것을 막기 위함이다.

### cell_synthesizer

`cell_synthesizer`또한 비슷하게 작성되었다.

````python
class CellAssignerState(AssignerState):
    cells: list[dict]

def cell_synthesizer(state: CellAssignerState):
    """번역된 ipynb 파일을 합성하는 함수"""

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
````

`cell_synthesizer`에서는 맨 앞 cell에 `KOREAN_TRANSLATED_MSG`를 넣어서 해당 문서가 번역되었음을 알려준다.

## 전체 그래프의 컴파일

전체 그래프를 컴파일하는 코드는 다음과 같다.

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

## 결론

이렇게 계층적 모델을 구성하면서 Langgraph를 사용하여 병렬적으로 문서를 번역하는 에이전트를 만들었다. 이를 통해 Langgraph의 강력한 기능을 활용하면서도 복잡한 문제를 해결할 수 있었다. 이제 이 에이전트를 통해 코드 저장소에 있는 문서들을 번역하는 작업을 수행할 수 있게 되었다.

## 전체 코드

````python
# LLM 초기화
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

#### 번역할 문서들의 루트 디렉터리와 제외할 디렉터리 목록을 설정 ####

root_dir = "../langgraph/"
excludes = [
    "../langgraph/docs/docs/reference",
    "../langgraph/docs/docs/index.md",
]

#####################################################################

GRAPH_OUTPUT_PATH = "agent.png"
LLM = "gpt-4o-mini"
KOREAN_TRANSLATED_MSG = "_한국어로 기계번역됨_\n\n"
TRANSLATE_PROMPT = "입력된 텍스트를 한국어로 '번역만' 수행합니다."

load_dotenv()  # load environment variables from `.env` file

llm = ChatOpenAI(model=LLM)

prograss_bar = None


class OverallState(TypedDict):
    """전체 상태 클래스"""

    root_dir: str  # 루트 디렉터리
    excludes: list[str]  # 제외할 디렉터리
    target_files_ext: list[str]  # 번역 대상 파일 확장자


class AssignerState(TypedDict):
    """할당 상태 클래스"""

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
    """번역 상태 클래스"""

    text: str
    type: Literal["markdown", "code"]


def set_save_path(dir: str, file: str) -> str:
    """저장 경로를 설정하는 함수"""
    return os.path.join(dir, file)


def is_translated_file(src_path: str) -> bool:
    """파일이 이미 번역되었는지 확인하는 함수"""

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
    """전체 프로세스를 관리하는 노드"""

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
        total=len(src_paths), desc="번역 진행도", position=0, leave=True
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
    """문자열의 왼쪽 공백을 반환하는 함수"""

    return text[: len(text) - len(text.lstrip())]


def get_r_whitespace(text: str) -> str:
    """문자열의 오른쪽 공백을 반환하는 함수"""

    return text[len(text.rstrip()) :]


IMG_REGEX = r"!\[.*\]\(data:image.*\)"


def text_assigner(state: TextAssignerState) -> Command[Literal["text_translator"]]:
    """텍스트를 청킹하여 Translator에 할당하는 함수"""

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
        ):  # 코드 블록이 시작되지만 100줄 안에 끝나지 않은 경우 -> 코드를 통째로 번역
            chunk = "```".join(text.split("```")[:2]) + "```"
        elif (
            chunk.count("```") % 2 == 1
        ):  # 코드 블록이 끝나지 않은 경우 -> 코드 시작 전까지 번역
            chunk = "```".join(chunk.split("```")[:-1])
        if match := re.search(IMG_REGEX, chunk):
            if match.start() == 0:  # 이미지가 첫 줄에 있는 경우
                chunk = chunk[: match.end()]
            else:  # 이미지가 중간에 있는 경우
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
    """ipynb 파일을 셀 단위로 나누어 Translator에 할당하는 함수"""

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
    """텍스트 조각을 번역하는 함수"""

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
    """번역된 문서를 합성하는 함수"""

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
    """번역된 ipynb 파일을 합성하는 함수"""

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

    # 그래프 실행
    graph.invoke(
        {
            "root_dir": root_dir,
            "excludes": excludes,
            "target_files_ext": [".md", ".ipynb"],
        }
    )
    prograss_bar.close()
````
