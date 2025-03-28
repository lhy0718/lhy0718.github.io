---
title: "[논문리뷰] Tree of Thoughts: Deliberate Problem Solving with Large Language Models (NeurIPS 2023)"
date: 2025-02-25 00:00:00 +0900
categories:
  - Paper Review
tags:
  - Reasoning
  - LLM
  - NLP
  - ToT
  - NeurIPS
---

요약: Chain-of-Thought(CoT) 접근법을 확장하여, LLM이 체계적인 문제 해결을 수행하도록 하는 Tree of Thoughts(ToT) 프레임워크를 제안한다.

---

# 1. Introduction

<img alt="image" src="https://github.com/user-attachments/assets/acb989e7-b1c0-4d34-b609-58de8d1f05db" />

- 기존의 대규모 언어 모델(LLM)은 일반적인 문제 해결을 수행할 수 있지만, 여전히 **토큰 단위의 좌→우 순차적 결정 과정**에 제한되며, **탐색, 전략적 예측(lookahead), 초기 결정의 중요성**이 요구되는 문제에서 한계를 보인다.

- 이를 해결하기 위해, 논문에서는 **Tree of Thoughts(ToT)**라는 새로운 프레임워크를 제안한다. ToT는 기존 **Chain of Thought(CoT)** 방식을 일반화하여, 문제 해결 과정에서 여러 개의 **중간 사고(thoughts)**를 탐색하고, 서로 다른 경로를 고려하며, 스스로 평가하여 최적의 결정을 내리도록 한다.

- 이 프레임워크를 통해 LLM은 다음과 같은 능력을 갖추게 된다:

  1. **여러 개의 사고 경로를 탐색**하여 최적의 문제 해결 방법을 찾는다.
  2. **자체 평가(self-evaluation)를 수행**하며 최상의 선택을 결정한다.
  3. **필요 시 되돌아가거나(Backtracking) 미래를 예측**하여(global decision-making) 더 나은 결정을 내린다.

- 실험 결과, ToT는 **수학 퍼즐(Game of 24), 창의적 글쓰기(Creative Writing), 미니 낱말 퍼즐(Mini Crosswords)** 등의 복잡한 문제 해결에서 기존 CoT 기반 방법보다 훨씬 높은 성능을 보인다.

  - 예를 들어, **Game of 24**에서는 기존 GPT-4의 CoT 프롬프트가 4%의 성공률을 기록한 반면, ToT 방식은 **74%의 성공률**을 달성한다.

- ToT는 LLM이 단순한 패턴 인식에 그치지 않고 **체계적 탐색과 계획을 수행할 수 있도록 지원**하여, **보다 강력한 문제 해결 능력을 갖춘 AI 시스템**을 구축하는 데 기여할 수 있다.

---

# 2. Background

- 기존의 대규모 언어 모델(LLM)은 다양한 문제 해결을 수행할 수 있지만, 여전히 **토큰 단위의 좌→우 순차적 생성 방식**에 의존하여 **체계적인 탐색, 전략적 예측, 되돌아가기(backtracking)** 등의 능력이 부족하다.

- 인간의 인지과학 연구에서는 사람들이 **빠르고 자동적인 직관적 사고(System 1)**와 **느리고 신중한 논리적 사고(System 2)**를 조합하여 문제를 해결한다고 본다.

  - 현재 LLM은 주로 **System 1** 방식에 가깝지만, **System 2** 방식의 **계획(plan), 탐색(search), 평가(evaluation)** 기능이 추가되면 더욱 강력한 문제 해결 능력을 갖출 수 있다.

- 기존의 LLM 기반 문제 해결 방식은 다음과 같다:

  1. **Input-Output (IO) Prompting**
     - 단순히 입력을 받아 출력을 생성하는 방식으로, 문제 해결 과정이 명시적으로 드러나지 않는다.
  2. **Chain of Thought (CoT) Prompting**
     - 문제 해결 과정을 여러 단계의 **중간 사고(thoughts)**로 분해하여 논리적 추론을 강화한다.
  3. **Self-Consistency with CoT (CoT-SC)**
     - 동일한 문제에 대해 여러 개의 CoT 경로를 생성하고, 가장 자주 등장하는 답변을 선택하여 신뢰도를 높인다.

- 하지만, 기존 방법들은 다음과 같은 한계를 가진다:
  - **CoT는 단일 경로로만 사고를 전개**하며, 여러 가지 가능한 해결 방법을 탐색하지 않는다.
  - **CoT-SC는 최종 답변에서만 다수결을 활용**할 뿐, **개별 사고 과정에서의 탐색 및 최적화가 부족**하다.
  - **계획, 되돌아가기, 평가를 수행하는 체계적인 탐색(search) 메커니즘이 필요**하다.

---

# 3. Tree of Thoughts: Deliberate Problem Solving with LM

- **Tree of Thoughts(ToT)**는 기존 LLM이 가진 한계를 극복하기 위해 설계된 새로운 문제 해결 프레임워크이다.
  - 기존 **CoT 방식은 단일 경로만 탐색**하며, 중간 사고(thoughts)를 생성할 때 다양한 대안을 고려하지 않는다.
  - 반면, **ToT는 여러 개의 사고 경로를 탐색**하고, 스스로 평가하며 최적의 경로를 선택하는 **계획적 문제 해결 방식**을 제공한다.

## 3.1 ToT의 핵심 아이디어

- 문제 해결을 **탐색(search) 과정으로 프레임화**하고, LLM이 여러 가지 해결 방법을 실험할 수 있도록 한다.
- 기존 AI 및 인지과학 연구에서 영감을 받아, **"문제를 해결하는 과정은 다양한 경로를 탐색하는 트리(tree) 구조"**와 같다고 가정한다.
- **ToT는 각 사고(thought)를 트리의 노드로 간주**하고, 다양한 사고 경로를 탐색하면서 최적의 해결책을 찾아간다.

## 3.2 ToT의 주요 구성 요소

1. **사고 단계(thought steps)의 분해**

   - CoT는 사고를 하나의 연속적인 텍스트로 생성하지만, ToT는 **작업의 특성에 맞게 사고 단계를 명확히 분해**하여 탐색 가능성을 높인다.

2. **사고 생성(Thought Generation, G)**

   - LLM이 주어진 상태에서 여러 개의 대체 사고(thoughts)를 생성할 수 있도록 한다.
   - 두 가지 방법을 사용:
     - **독립 샘플링(Independent Sampling)**: 서로 독립적인 사고를 여러 개 생성(CoT 기반).
     - **연속적 제안(Sequential Proposing)**: 이전 사고를 기반으로 단계적으로 새로운 사고를 제안.

3. **상태 평가(State Evaluation, V)**

   - 여러 개의 사고 경로 중 **가장 유망한 경로를 선택**하기 위해, LLM이 스스로 평가를 수행한다.
   - 평가 방법:
     - **개별 평가(Individual Evaluation)**: 각 사고를 독립적으로 평가(예: 점수 부여).
     - **비교 평가(Comparative Voting)**: 여러 사고 중 최적의 것을 선택하는 다중 선택 방식.

4. **탐색 알고리즘(Search Algorithm)**
   - 트리 구조를 탐색하는 방법으로, **너비 우선 탐색(BFS)과 깊이 우선 탐색(DFS)**을 활용한다.
   - 문제의 특성에 따라 적절한 탐색 방식을 선택:
     - **BFS (너비 우선 탐색)**: 여러 경로를 동시에 탐색하며 최적의 경로를 찾는다.
     - **DFS (깊이 우선 탐색)**: 하나의 유망한 경로를 깊게 탐색하되, 필요할 경우 가지치기를 수행하여 비효율적인 경로를 제거.

<img alt="image" src="https://github.com/user-attachments/assets/0432cc84-f329-44e4-9a72-478a3795140d" />

## 3.3 ToT의 장점

- **일반성(Generality)**
  - 기존 **IO Prompting, CoT, CoT-SC, Self-Refinement** 등의 접근법을 포괄하는 개념이다.
- **모듈성(Modularity)**
  - 사고의 분해, 생성, 평가, 탐색 알고리즘을 독립적으로 조합하여 사용할 수 있다.
- **적응성(Adaptability)**
  - 다양한 문제 유형에 맞춰 탐색 및 평가 전략을 변경할 수 있다.
- **편의성(Convenience)**

  - 추가적인 학습 없이 **사전 학습된 LLM을 프롬프트만으로 활용**할 수 있다.

- ToT는 **LLM이 더욱 강력한 문제 해결 능력을 가질 수 있도록 설계된 프레임워크**이며, 이후 실험을 통해 다양한 문제에서 기존 방법보다 우수한 성능을 보인다는 점을 입증한다&#8203;:contentReference[oaicite:0]{index=0}.

---

# 4. Experiments

<img alt="image" src="https://github.com/user-attachments/assets/8d883e2a-c550-4e9b-bb44-71adb7eb671f" />

- 논문에서는 **Tree of Thoughts(ToT)**가 기존 방법보다 뛰어난 문제 해결 능력을 갖추었음을 입증하기 위해 세 가지 복잡한 문제를 실험한다.
- 실험 대상 문제:
  1. **Game of 24** (수학적 추론)
  2. **Creative Writing** (창의적 글쓰기)
  3. **Mini Crosswords** (단어 퍼즐)

## 4.1 Game of 24 (수학적 문제 해결)

<img alt="image" src="https://github.com/user-attachments/assets/2809ef6c-5ec2-4dda-b5ff-b1614b87c61d" />

- **문제 설명**

  - 주어진 네 개의 숫자(예: 4, 9, 10, 13)를 **사칙연산을 활용해 24를 만드는 방정식을 찾는 문제**이다.

- **실험 설정**

  - 난이도가 높은 문제 100개를 선택하여 테스트.
  - 기존 방법들과 비교:
    - **IO Prompting**: 입력을 받고 바로 정답을 출력하는 방식.
    - **CoT Prompting**: 중간 사고 단계를 포함하여 문제 해결.
    - **CoT-SC (Self-Consistency)**: 여러 개의 CoT 답안을 생성한 후 다수결 선택.
    - **IO + Refine**: 잘못된 답안을 수정하면서 점진적으로 개선.

- **ToT 적용 방법**

  - 문제 해결을 **트리 탐색 문제로 변환**하여 해결.
  - 각 단계에서 여러 개의 연산을 탐색하고, **가장 가능성 높은 경로를 선택**하여 진행.

- **결과 비교**

  | 방법               | 성공률 |
  | ------------------ | ------ |
  | IO Prompting       | 7.3%   |
  | CoT Prompting      | 4.0%   |
  | CoT-SC (k=100)     | 9.0%   |
  | ToT (b=1)          | 45%    |
  | ToT (b=5)          | 74%    |
  | IO + Refine (k=10) | 27%    |
  | IO (best of 100)   | 33%    |
  | CoT (best of 100)  | 49%    |

  - **ToT(b=5)는 74% 성공률을 기록**, 기존 방법보다 월등히 높은 성능을 보임.

---

## 4.2 Creative Writing (창의적 글쓰기)

<img alt="image" src="https://github.com/user-attachments/assets/04b65615-7240-4313-9631-7e9c4fef06cf" />

- **문제 설명**

  - 주어진 **4개의 문장**을 포함하는 **일관성 있는 4단락 글쓰기**를 수행하는 문제.

- **실험 설정**

  - 100개의 무작위 문장을 사용하여 테스트.
  - 평가 기준:
    - GPT-4를 활용한 자동 평가(1~10점)
    - 사람을 대상으로 한 블라인드 비교 평가

- **ToT 적용 방법**

  - ToT는 **우선적으로 글의 구조를 계획한 후 본문을 작성**하는 방식을 사용.
  - **두 단계 트리 탐색**
    1. 여러 개의 글쓰기 계획(5개) 생성 후 **투표로 최적의 계획 선택**
    2. 선택된 계획을 바탕으로 여러 개의 본문(5개) 작성 후 **투표로 최적의 본문 선택**

- **결과 비교**

  | 방법          | GPT-4 평가 (1~10점) | 인간 평가 비교 |
  | ------------- | ------------------- | -------------- |
  | IO Prompting  | 6.19                | -              |
  | CoT Prompting | 6.93                | 21% 우세       |
  | ToT           | 7.56                | 41% 우세       |

  - **ToT 방식이 가장 높은 평가(7.56점)를 받으며, 인간 평가에서도 가장 선호됨.**

---

## 4.3 Mini Crosswords (단어 퍼즐)

<img alt="image" src="https://github.com/user-attachments/assets/cad40a56-01c0-4e05-b18d-3f51f5e93e19" />

- **문제 설명**

  - 5×5 낱말 퍼즐을 풀어 **주어진 단어 힌트에 맞는 정답을 채우는 문제**.

- **실험 설정**

  - 주어진 힌트에 맞춰 정답을 완성하는 정확도를 측정.

- **ToT 적용 방법**

  - 각 단어를 채울 때, **가능한 여러 개의 후보를 생성하고, 가장 적절한 단어를 선택하는 방식**을 사용.
  - 기존 방법(CoT)보다 다양한 정답 가능성을 평가하는 것이 특징.

- **결과 비교**
  - ToT는 **기존 GPT-4보다 정답률이 높고, 단어 배열의 일관성이 우수함**을 보임.

---

## 4.4 결론

- ToT는 **체계적인 탐색 및 평가 전략을 적용하여** 기존 방법보다 훨씬 높은 성능을 기록한다.
- 특히 **수학적 추론(Game of 24)**과 **창의적 문제 해결(Creative Writing, Mini Crosswords)**에서 **기존 CoT보다 우수한 결과**를 보인다.
- **기존 LLM 접근법(IO, CoT, CoT-SC 등)과 달리, ToT는 트리 탐색을 통해 다양한 해결 경로를 고려하며 최적의 해결책을 찾는다**.

---

# 5. Related Work

- 본 논문은 기존 **언어 모델 기반 문제 해결 기법**을 확장하여 **Tree of Thoughts(ToT)** 프레임워크를 제안하며, 여러 연구 분야에서 영향을 받았다.

## 5.1 Chain of Thought (CoT) Prompting

- **CoT(Chain of Thought) 프롬프트**는 언어 모델이 복잡한 문제를 단계별로 해결할 수 있도록 하는 대표적인 방법이다.
- LLM이 최종 답변을 바로 생성하는 대신 **중간 사고 과정(thoughts)**을 명시적으로 표현하여 더 정확한 추론을 수행할 수 있도록 한다.
- 그러나 CoT는 **단일 경로에서만 사고를 전개**하며, 다양한 가능성을 탐색하거나 평가하는 기능이 부족하다.
- ToT는 이러한 한계를 극복하기 위해 **여러 사고 경로를 탐색하고 평가하는 메커니즘**을 추가한다.

## 5.2 Self-Consistency in CoT

- **CoT-SC(Self-Consistency)** 기법은 같은 문제에 대해 여러 개의 CoT 답안을 생성한 후, **가장 빈번하게 등장하는 정답을 선택**하는 방식이다.
- 다수결 방식을 통해 정답의 신뢰도를 높이지만, **각 사고 단계에서 대체 경로를 탐색하는 기능이 부족**하다.
- ToT는 CoT-SC의 다중 샘플링 개념을 확장하여, **단순한 다수결이 아닌 탐색과 평가를 결합한 문제 해결 방식을 적용**한다.

## 5.3 Tool Use by Language Models

- 최근 연구에서는 LLM이 외부 도구(계산기, API 등)를 활용하여 문제 해결 능력을 확장할 수 있음을 보였다.
- 예: **Toolformer** 연구는 LLM이 외부 도구를 스스로 호출하여 문제를 해결하는 방법을 탐구했다.
- ToT 역시 LLM이 **자체적인 평가 및 탐색 능력을 갖추도록 하여, 단순한 정답 생성이 아니라 깊이 있는 문제 해결을 수행**하도록 한다.

## 5.4 Search Algorithms and AI Planning

- **탐색(Search) 및 계획(Planning) 기법**은 전통적인 AI 문제 해결의 핵심 요소이다.
- 예: **A\* 알고리즘, 몬테카를로 트리 탐색(MCTS)** 등의 기법은 게임 및 최적화 문제에서 사용된다.
- ToT는 이러한 탐색 기법을 LLM에 적용하여, 문제 해결 과정에서 **여러 해결 경로를 평가하고 최적의 답을 선택하는 방식**을 제공한다.

## 5.5 Human Problem-Solving Theories

- 심리학 연구에서는 **인간이 문제를 해결할 때 트리 구조의 탐색을 수행**한다는 점을 강조해 왔다.
- 예: **Newell & Simon(1950s)의 문제 해결 이론**에서는 **상태 공간 탐색(state-space search)** 개념을 활용하여 인간의 문제 해결 방식을 설명하였다.
- ToT는 이러한 인지과학적 이론을 반영하여, **LLM이 사고 과정을 보다 체계적으로 탐색하고 평가할 수 있도록 설계**되었다.

## 5.6 결론

- ToT는 **CoT 및 기존 LLM 기반 문제 해결 기법의 한계를 보완**하며, **전통적인 탐색 알고리즘 및 인지과학적 문제 해결 이론을 접목**하여 더욱 강력한 문제 해결 능력을 갖춘다.
- 기존 연구들이 **단일 경로 기반의 문제 해결**에 초점을 맞췄다면, ToT는 **트리 구조를 활용하여 여러 경로를 탐색하고 최적의 답을 선택하는 방식을 도입**한다.

---

# 6. Discussion

- 본 논문에서는 **Tree of Thoughts(ToT)** 프레임워크를 통해 **LLM의 문제 해결 능력을 향상시키는 방법**을 탐구하였다.
- 본 장에서는 ToT의 강점, 한계점, 향후 연구 방향을 논의한다.

## 6.1 ToT의 주요 이점

1. **더 나은 문제 해결 능력**

   - ToT는 **단순한 답변 생성이 아니라 탐색 기반 문제 해결을 수행**하며, 기존 방법(CoT, CoT-SC)보다 **더 높은 성공률을 기록**한다.
   - 다양한 사고 경로를 탐색하고, 평가를 통해 최적의 경로를 선택함으로써 **보다 신뢰할 수 있는 답변을 생성**한다.

2. **모듈성과 확장성**

   - ToT는 **다양한 문제 유형에 적용 가능**하며, **사고 분해, 생성, 평가, 탐색 알고리즘을 독립적으로 조합**할 수 있다.
   - 기존 LLM 기반 접근법과도 쉽게 결합할 수 있어 **유연한 문제 해결 프레임워크로 활용 가능**하다.

3. **추론 과정의 명확성**
   - 기존 LLM은 "블랙박스" 문제를 가진 반면, ToT는 **각 단계에서 어떠한 사고 과정이 진행되는지 명확하게 설명할 수 있다.**
   - 이로 인해 **AI의 추론 과정이 더 투명해지고, 인간이 신뢰할 수 있는 모델 설계가 가능**해진다.

---

## 6.2 ToT의 한계 및 과제

1. **계산 비용 증가**

   - ToT는 기존 CoT보다 **여러 사고 경로를 탐색해야 하므로 계산 비용이 증가**할 수 있다.
   - 최적의 탐색 전략을 설계하여 **성능을 유지하면서도 연산량을 줄이는 연구가 필요**하다.

2. **탐색 알고리즘의 최적화 필요**

   - 현재는 BFS, DFS 등의 단순 탐색 알고리즘을 사용하지만, **더 정교한 탐색 방법(A\*, MCTS 등)을 적용하면 성능 향상이 가능**할 것이다.
   - 특히, **문제 유형에 맞는 최적의 탐색 전략을 자동으로 선택하는 연구**가 필요하다.

3. **사고 평가의 신뢰성 문제**
   - ToT는 LLM이 자체 평가를 수행하지만, **평가 방식이 완벽하지 않아 부정확한 판단을 내릴 가능성**이 있다.
   - 이를 해결하기 위해 **보다 정교한 평가 기법을 도입하거나, 인간 피드백을 활용하는 방법**을 고려할 수 있다.

---

## 6.3 향후 연구 방향

1. **탐색 최적화 연구**

   - 다양한 탐색 알고리즘(A\*, MCTS 등)과 ToT를 결합하여 **보다 효율적인 문제 해결 기법을 개발**할 필요가 있다.

2. **자기 지도 학습(Self-Supervised Learning) 적용**

   - ToT가 특정 유형의 문제를 해결할 때 **스스로 학습하면서 점점 더 나은 탐색 전략을 찾을 수 있도록 개선**할 수 있다.

3. **다중 모달 문제 해결로 확장**

   - 현재 ToT는 언어 모델에 초점이 맞춰져 있지만, **이미지, 코드, 수식 등의 다중 모달 데이터를 활용하는 방향으로 확장할 수 있다.**
   - 예를 들어, **ToT를 활용하여 이미지 기반 퍼즐을 해결하거나, 프로그래밍 문제를 푸는 방식으로 발전 가능**하다.

4. **인간과의 협업을 고려한 ToT 설계**
   - ToT를 **인간이 직접 수정하거나 피드백을 줄 수 있는 시스템과 결합하면 더욱 강력한 문제 해결 도구가 될 수 있다.**
   - 예를 들어, **인간이 특정 사고 경로를 직접 선택하거나 평가하도록 설계하는 연구**가 가능하다.

---

## 6.4 결론

- ToT는 **기존 LLM의 단순한 순차적 사고 방식을 극복하고, 체계적인 탐색과 평가를 수행하는 새로운 문제 해결 프레임워크**를 제시한다.
- **더 높은 문제 해결 성능, 모듈성과 확장성, 투명한 추론 과정** 등 여러 장점을 가지며, **향후 연구를 통해 더욱 정교한 탐색 및 평가 메커니즘을 개발할 필요가 있다**.

---

# 독자 의견

- 한계점 파트에서 언급한 것 처럼 문제 유형에 대한 최적의 탐색전략을 찾는 것이 필요하다. 또한 어떤 간단한 문제에 있어서는 ToT까지도 필요 없이 CoT 혹은 그 이전 단의 추론만으로도 충분할 수 있다. 이를 판별하여 최적의 솔루션을 찾는것은 ToT로 인한 추론 비용 증가를 방지하는데 도움이 될 것이다.
- 이러한 매타 탐색은 LLM를 사용할수도 있고, 아니면 SLM을 사용해서 CoT, CoT-SC, ToT 등의 탐색 방법들 중 최적의 방식을 먼저 선택하는 방식으로 이루어질 수 있을 것이다.
