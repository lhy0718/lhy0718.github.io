---
title: "[Langgraph] Implementing a ReAct agent"
date: 2025-02-22 15:00:00 +0900
categories:
  - Agents
tags:
  - Langgraph
  - Langchain
  - Agents
  - LLM
---

Summary: Implementing a ReAct Agent with Langgraph

---

# Overview

[ReAct](https://arxiv.org/abs/2210.03629) is a methodology that enables LLMs to interact with environments (such as the web), helping to overcome hallucination and error propagation issues commonly found in traditional CoT (Chain of Thought) approaches. Langgraph incorporates a pre-built ReAct agent influenced by this paper, which is explained in detail [here](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/).

---

# Implementing a ReAct Agent

A `tool` is defined using **input arguments** and a **docstring**, which guide the LLM on how to utilize the tool effectively. The **input arguments must be of type `string`, and the docstring is mandatory**.

Example:

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

This tool receives `nyc` or `sf` as input from the LLM and returns the corresponding weather information. The input values are constrained using `Literal`, and the docstring helps the LLM determine the appropriate input. (The internal mechanics of how this works are unclear.)

Another example:

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

This tool takes `a` and `b` as inputs from the LLM and returns the product of `a` and the maximum value in `b`. The `Annotated` type adds descriptions to the input arguments.

Additionally, the schema of the tool's arguments can be inspected using the following code:

```python
print(multiply_by_max.args_schema.model_json_schema())
```

Now, let's create an agent that can either provide weather information or perform mathematical calculations based on user input. This can be easily implemented using `langgraph.prebuilt.create_react_agent`.

```python
from typing import Literal, Annotated, List
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()  # Load API keys from the .env file

llm = ChatOpenAI(model="gpt-4o-mini")  # Initialize the LLM

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

---

# The Graph Structure

The graph can be visualized using the following code:

```python
from IPython.display import Image, display

display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

![image](https://github.com/user-attachments/assets/d677b6df-ff11-463b-8a24-f9f6f9d70f72)

Now, let's test the agent to see how the tools operate.

---

# Using the ReAct Agent

```python
from langchain_core.messages import HumanMessage

# Function to display streamed output in a clean format
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


msg = {"messages": HumanMessage("What's the weather in nyc?")}  # User input
print_stream(graph.stream(msg, stream_mode="values"))  # Invoke the graph
```

## Output:

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

Now, let's test the mathematical computation tool.

```python
msg = {"messages": HumanMessage("Multiply 3 by the maximum of 1, 2, 9, 4, 5")}
print_stream(graph.stream(msg, stream_mode="values"))
```

## Output:

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

---

# Conclusion

Using **Langgraph** and **Langchain’s tool system**, implementing a ReAct agent is remarkably simple and efficient.
