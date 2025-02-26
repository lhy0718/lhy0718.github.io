---
title: "[Langgraph] 이 에이전트는 10원으로 논문 요약해줍니다."
date: 2025-02-26 22:00:00 +0900
categories:
  - Agents
tags:
  - Langgraph
  - Langchain
  - Agents
  - LLM
---

요약: Langgraph를 사용하여 논문을 자동으로 원하는 언어로 요약해주는 에이전트를 만들어보자.

---

# 개요

Langgraph 사용법을 익히기 위해 작은 에이전트들을 제작하고 있던 중에 논문 요약 에이전트를 만들어보았다. 이 에이전트는 논문을 PDF 형식으로 입력받아 원하는 언어로 요약해주는 기능을 제공한다.

그래프의 전체적인 구조는 다음과 같다.

```
[PDF 읽기 및 섹션 나누는 노드] -> [요약 노드 여러 개] -> [결과 병합 및 저장 노드]
```

다음 장에서는 PDF 읽기 및 섹션 나누는 노드에서 PDF를 파싱하는 방법에 대해 설명하겠다.

# LLM 정의

이 에이전트에서는 GPT-4o-mini를 사용할 것이다. 이를 위해 다음과 같이 LLM을 정의하였다.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv() # .env 파일에서 OpenAI, LangSmith 등의 API 키를 불러온다.

llm = ChatOpenAI(model="gpt-4o-mini")
```

# PDF 파싱하기

Langgraph는 PDF뿐 아니라 다양한 형식의 파일을 로딩하는 [기능](https://python.langchain.com/docs/integrations/document_loaders/)을 제공한다. PDF를 로딩하고 파싱해주는 로더들도 [다양하게 준비되어 있는데,](https://python.langchain.com/docs/integrations/document_loaders/#pdfs) 나는 이 중에서 `PyPDFLoader`를 사용하여 에이전트를 구현했다.

먼저 파일을 로딩하는 부분의 코드는 다음과 같다. (state 변수는 아래에서 설명하겠다.)

```python
loader = PyPDFLoader(state["file_path"])
pages = []
async for page in loader.alazy_load():
  pages.append(page)
content = "\n".join([page.page_content for page in pages])
```

`async` 함수를 사용하기 때문에 해당 노드의 상위 함수에 `async` 키워드를 붙여주어야 한다. 또한 나중에 graph를 실행할 때 `invoke` 대신에 `ainvoke`를 사용해야 하는 것도 주의해야 한다. 이는 또한 아래에서 설명하겠다.

# section_spliter (PDF 파싱 및 섹션 나누는 노드)

노드를 정의하기 전에 노드가 사용할 State를 먼저 정의해야한다. 필요한 변수는 요약할 PDF의 파일 경로(`file_path`), 요약에 사용될 언어(`language`), 요약 결과를 저장할 변수(`summary`)이다.

```python
class OverallState(TypedDict):
    """전체 상태 클래스"""

    file_path: str
    language: str
    summary: Annotated[list, operator.add]
```

`summary` 변수는 각 섹션을 요약하는 노드들이 출력하는 요약 결과를 [map-reduce](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.ipynb) 방식으로 합치기 위해 저 형식으로 정의하였다.

다음은 노드 안에서 PDF를 읽어서 섹션으로 나누는 코드이다. 이를 위해 LLM을 사용하여 `content`에서 목차를 추출한 뒤에, 목차를 가지고 `content`를 나눌 것이다. 이 과정은 다음과 같이 구현할 수 있다.

```python
res = llm.invoke(
    [
        SystemMessage(
            "Acknowledgements, Appendix 제외한 논문의 최상위 목차를 한 줄씩 출력합니다.\n출력 포맷: 1 Introduction\n2 ..."
        ),
        HumanMessage(content),
    ]
)
```

그럼 논문의 목차가 뽑힐 것이다. 그런데 나는 abstract를 목차에 넣고 싶고, 마지막 목차와 reference 사이를 구분할 수 있는 방법이 필요했다. 이를 위해 다음과 같이 목차를 처리하고 섹션을 나누었다.

```python
titles = [line.strip() for line in res.content.splitlines()]
if "Abstract" not in titles:
    titles.insert(0, "Abstract") # Abstract가 목차에 없으면 추가
if "References" not in titles:
    titles.append("References") # References가 목차에 없으면 추가

sections = []
for i, title in enumerate(titles[:-1]): # 마지막 목차는 References이므로 제외
    section = content.split(title)[1].split(titles[i + 1])[0] # 목차 사이의 내용을 추출
    sections.append(section)
```

목차 리스트는 LLM이 뽑는 것이라 정확하지 않을 수 있지만, 대부분의 경우에 잘 동작하므로 큰 문제는 없다.

이제 해당 노드의 전체 코드는 다음과 같다.

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
                "Acknowledgements, Appendix 제외한 논문의 최상위 목차를 한 줄씩 출력합니다.\n출력 포맷: 1 Introduction\n2 ..."
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

노드에서 중요하게 보아야 하는 부분은 `-> Command[Literal["section_summarizer"]]:` 으로 노드의 리턴값을 명시해주었다는 것이다. 이는 리턴값이 `Command`타입이기 때문이고, `Command`를 리턴할때는 그래프의 edge를 그래프 빌더에 전달하지 않기 때문이다.
그리고 `goto`에는 다음 노드의 이름과 해당 노드로 제목과 섹션, 언어를 전달한다.

# section_summarizer (요약 노드)

section_summarizer는 LLM을 사용하여 전달받은 각 섹션을 요약한다. 이 노드에서는 title로 "Abstract"를 받으면 한 두줄로 요약을 해주고, 그 외의 경우에는 섹션을 제한 없이 요약해준다. 또한 요약결과는 markdown 형식을 따르도록 하였다. 노드의 전체 코드는 아래와 같다.

```python
class SectionState(TypedDict):
    """Section 상태 클래스"""

    title: str
    section: str
    language: str

def section_summarizer(state: SectionState):

    prompt = ""
    if state["title"] == "Abstract":
        prompt = f"논문의 초록을 1~2줄로 {state['language']}로 요약"
    else:
        prompt = f"Section '{state['title']}'의 내용을 .md 포맷으로 {state['language']}로 요약 (제목은 '# {state['title']}' 으로 시작). 수식 블록은 $$로 감싸기."

    res = llm.invoke(
        [
            SystemMessage(prompt),
            HumanMessage(state["section"]),
        ]
    )
    return {"summary": [res.content]}
```

# gatherer (결과 병합 및 저장)

gatherer에서는 `state["summary"]`에 병합된 결과들을 후처리하고 .md파일로 저장한다. 이 노드의 전체 코드는 다음과 같다.

````python
def gatherer(state: OverallState):
    summaries = []
    for section in state["summary"]:
        section = section.strip()
        if section.startswith("```markdown"):
            section = section.split("```")[1]
        if section.startswith("```"):
            section = section.split("```")[1]
        if section.endswith("```"):
            section = section.rsplit("```")[0]
        summaries.append(section)

    total_summary = "---\n\n" + "\n\n---\n\n".join(state["summary"])

    with open(f"{os.path.splitext(state['file_path'])[0]}.md", "w") as f:
        f.write(total_summary)
````

위 코드에서 2 가지의 후처리를 수행하고 있다. 하나는 markdown 코드 블록 형식으로 나온 LLM 출력의 코드 블록을 없애주는 것이고, 다른 하나는 섹션 간에 구분선을 넣어주는 것이다. 그리고 나서는 원래 파일의 이름을 가져와서 .md 확장자를 붙여서 저장한다.

# 그래프 빌드 및 실행

이제 각 노드들을 그래프로 묶어보자. 그래프를 빌드하는 코드는 다음과 같다.

```python
graph_builder = StateGraph(OverallState)
graph_builder.add_node("section_spliter", section_spliter)
graph_builder.add_edge(START, "section_spliter")
graph_builder.add_node("section_summarizer", section_summarizer)
graph_builder.add_node("gatherer", gatherer)
graph_builder.add_edge("section_summarizer", "gatherer")
graph_builder.add_edge("gatherer", END)
graph = graph_builder.compile()
```

이를 아래 코드로 시각화하면 다음과 같이 보인다.

```python
from IPython.display import Image, display

display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

![output](https://github.com/user-attachments/assets/17b38111-7202-49f0-a26b-86202eadfefb)

실행은 다음과 같이 한다. 맨 처음의 노드가 async 함수이므로 `ainvoke`를 사용해야 한다.

```python
await graph.ainvoke({"file_path": "VaRMI.pdf", "language": "한국어"})
```

이제 논문이 요약된 .md 파일이 생성되었을 것이다. 만약 에러가 나온다면 LLM이 목차를 추출하는 부분에서 오류가 난 것이므로 다시 실행하면 된다.

# 결론

이렇게 Langgraph를 사용하여 논문을 요약하는 에이전트를 만들어보았다. 이 에이전트를 사용하여 몇 개의 논문을 요약해보았는데, 10원 내외의 API 요금이 사용된다. 아주 저렴하고 빠르게 논문을 요약하여 읽고 싶을 때 유용하게 사용할 수 있을 것이다.

# 전체 코드

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
    """전체 상태 클래스"""

    file_path: str
    language: str
    summary: Annotated[list, operator.add]


class SectionState(TypedDict):
    """Section 상태 클래스"""

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
                "Acknowledgements, Appendix 제외한 논문의 최상위 목차를 한 줄씩 출력합니다.\n출력 포맷: 1 Introduction\n2 ..."
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
        prompt = f"논문의 초록을 1~2줄로 {state['language']}로 요약"
    else:
        prompt = f"Section '{state['title']}'의 내용을 .md 포맷으로 {state['language']}로 요약 (제목은 '# {state['title']}' 으로 시작). 수식 블록은 $$로 감싸기."

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
            section = section.split("```")[1]
        if section.startswith("```"):
            section = section.split("```")[1]
        if section.endswith("```"):
            section = section.rsplit("```")[0]
        summaries.append(section)

    total_summary = "---\n\n" + "\n\n---\n\n".join(state["summary"])
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

await graph.ainvoke({"file_path": "VaRMI.pdf", "language": "한국어"})
````
