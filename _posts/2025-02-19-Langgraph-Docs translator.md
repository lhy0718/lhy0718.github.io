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
