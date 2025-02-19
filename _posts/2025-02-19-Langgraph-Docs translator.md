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

이제 각 노드의 코드를 설명하겠다. 
