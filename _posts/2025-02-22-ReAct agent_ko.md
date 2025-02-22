---
title: "[Langgraph] ReAct agent 구현해보기"
date: 2025-02-22 00:00:00 +0900
categories:
  - Agents
tags:
  - Langgraph
  - Agents
  - LLM
---

요약: ReAct Agent를 Langgraph로 구현해본다.

---

# 개요

[ReAct](https://arxiv.org/abs/2210.03629)는 LLM이 환경(웹과 같은)과 상호작용함으로써 기존의 CoT에서 일어나는 환각과 오류전파 문제를 극복하는 방법론이다. Langgraph에서는 해당 논문의 영향을 받아서 미리 구축된 ReAct를 사용하는 기능을 추가했고, [여기서](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) 설명하고 있다.

# ReAct agent 구현

`tool`은 **입력 인수**, **docstring** 을 통해 LLM이 어떻게 해당 tool을 활용해야 하는지를 정의한다. **tool의 입력 인수는 string type이고, docstring은 필수이다.**

예시:

```python
from typing import Literal
from langchain_core.tools import tool

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")
```

이 tool은 LLM으로부터 nyc 또는 sf를 입력받아서 해당 도시의 날씨 정보를 반환한다. 입력되는 값은 `Literal`을 사용하여 제한되고, 입력을 결정하는 것은 docstring을 통해 LLM이 결정하게 되는 듯 하다. (내부 작동 방식은 모르겠다.)

또 다른 예시:

```python
from typing import Annotated, List
from langchain_core.tools import tool

@tool
def multiply_by_max(
    a: Annotated[int, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)
```

이 tool은 LLM으로부터 `a`와 `b`를 입력받아서 `a`와 `b`의 최대값을 곱한 값을 반환한다. `Annotated`를 사용하여 입력의 설명을 추가할 수 있다.

또한 tool이 만든 인수의 스키마를 아래 코드를 통해 확인할 수 있다.

```python
print(multiply_by_max.args_schema.model_json_schema())
```

두 개의 tool을 묶어서 사용자 입력에 대해 날씨 정보를 알려주거나, 수학계산을 하는 에이전트를 만들어보자. 이는
langgraph.prebuilt.create_react_agent를 사용하여 쉽게 구현할 수 있다.

```python
from typing import Literal, Annotated, List
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


load_dotenv()  # .env 파일에서 비밀키 값 로드

llm = ChatOpenAI(model="gpt-4o-mini")  # LLM 초기화

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

@tool
def multiply_by_max(
    a: Annotated[int, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)

tools = [get_weather, multiply_by_max]

graph = create_react_agent(model, tools=tools)
```

그래프 모양은 이렇다.

그럼 사용을 해보며 tool이 어떻게 작동하는지 확인해보자.

```python
from langchain_core.messages import HumanMessage

# 스트림 출력을 예쁘게 보여주는 함수
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


msg = {"messages": HumanMessage("What's the weather in nyc?")} # 사용자 입력
print_stream(graph.stream(msg, stream_mode="values")) # 그래프 호출
```

출력:

```text
================================ Human Message =================================

What's the weather in nyc?
================================== Ai Message ==================================
Tool Calls:
get_weather (call_f72kfLYwXZhyNdiimAGjuYG4)
Call ID: call_f72kfLYwXZhyNdiimAGjuYG4
Args:
city: nyc
================================= Tool Message =================================
Name: get_weather

It might be cloudy in nyc
================================== Ai Message ==================================

The weather in NYC might be cloudy.
```

이번에는 수학 계산을 해보자.

```python
msg = {"messages": HumanMessage("Multiply 3 by the maximum of 1, 2, 9, 4, 5")}
print_stream(graph.stream(msg, stream_mode="values"))
```

출력:

```text
================================ Human Message =================================

Multiply 3 by the maximum of 1, 2, 9, 4, 5
================================== Ai Message ==================================
Tool Calls:
  multiply_by_max (call_9LniwXXq3XDPhdN1ALliL4hJ)
 Call ID: call_9LniwXXq3XDPhdN1ALliL4hJ
  Args:
    a: 3
    b: [1, 2, 9, 4, 5]
================================= Tool Message =================================
Name: multiply_by_max

27
================================== Ai Message ==================================

The result of multiplying 3 by the maximum of 1, 2, 9, 4, and 5 is 27.
```

# 결론

Langgraph와 Langchain의 tool을 사용하면 놀랍도록 쉽게 ReAct Agent를 구현할 수 있다.
