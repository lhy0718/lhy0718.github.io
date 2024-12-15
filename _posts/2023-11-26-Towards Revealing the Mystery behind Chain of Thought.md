---
title: "Towards Revealing the Mystery behind Chain of Thought - A Theoretical Perspective (NeurIPS 2023) -- Part 1"
date: 2023-11-26 00:00:00 +0900
categories:
  - Paper Review
tags:
  - NeurIPS
  - Chain of Thought
  - Large Language Model
---

요약: 이 논문은 CoT가 LLM의 성능을 향상시키는 이론적인 이해를 제시한다. 이 논문은 circuit complexity theory를 사용하여 bounded-depth Transformers가 수학/산술문제의 인풋 길이에 대해 super-polynomially하게 커지지 않는 이상 문제를 해결하기 어렵다는 것을 입증하고, 반면 autoregressive Transformers는 상수 크기로도 CoT를 사용하면 문제를 해결할 수 있음을 입증한다. 또한 CoT가 있는 LLM이 동적 프로그래밍으로 알려진 일반적인 의사 결정 문제를 처리할 수 있다는 것을 보여준다.

## Paper

[Towards Revealing the Mystery behind Chain of Thought - A Theoretical Perspective](https://arxiv.org/pdf/2305.15408)

## 사전지식

- 계산 복잡도 이론에 대한 전반적인 이해
- **Circuit complexity** (회로 복잡도): [https://en.wikipedia.org/wiki/Circuit_complexity](https://en.wikipedia.org/wiki/Circuit_complexity)

## Abstract

- CoT는 LLM의 성능을 높인다는 것이 경험적으로 입증됨
- 하지만 CoT의 역할에 대한 이론적인 이해는 어려움
- circuit complexity theory를 사용하여 bounded-depth Transformers가 수학/산술문제의 인풋 길이에 대해 super-polynomially하게 커지지 않는 이상 문제를 해결하기 어렵다는 것을 입증
- 반면 autoregressive Transformers는 상수 크기로도 CoT를 사용하면 문제를 해결할 수 있음을 입증
- CoT가 있는 LLM이 동적 프로그래밍으로 알려진 일반적인 의사 결정 문제를 처리할 수 있다는 것을 보여줌

## 1. **Introduction**

- Autoregressive Transformer LLM은 prompt를 통해 어떤 테스크던 간에 테스크 설명을 통해 추론이 가능함
- CoT는 추론의 중간과정을 출력하게 함으로써 LLM의 능력을 향상 시키지만, 어떤 매커니즘을 통해 성능 향상이 발생하는지는 미스터리함
- 본 논문에서는 두 가지 단순한 수학 문제에 대한 LLM의 능력을 보는 것 부터 시작함
  - evaluating arithmetic expressions
  - solving linear equations
- CoT를 사용하지 않는 bounded-depth Transformer model은 인풋 길이에 대해 super-polynomially 하게 모델의 크기가 커져도 간단한 수학문제를 해결하지 못함을 입증 (Theorems 3.1 and 3.2)
  - 이는 parallel complexity 때문
- CoT를 사용하는 autoregressive Transformer model은 constant size의 모델로도 간단한 수학문제를 해결할 수 있음을 입증 (Theorems 3.3 and 3.4)
- 또한 본 논문에서는 동적프로그래밍(DP)을 해결하는 CoT의 능력에 대해 분석함 (Theorem 4.7)
  - 또한 polynomial size의 bounded-depth Transformers는 Context-Free Grammar Membership Testing으로 알려진 DP문제를 해결하지 못함을 입증 (Theorem 4.8)
- 여러 DP 문제에 대한 실험도 진행
  - longest increasing subsequence (LIS), edit distance (ED) 문제에 대한 실험
  - CoT가 사용되지 않으면 성능이 떨어짐
  - CoT를 사용하는 경우에 긴 인풋에 대해 더 잘 일반화 되므로, 모델이 단순히 인풋-아웃풋 분포를 암기하는 것이 아니라 실제로 추론을 하고 있음을 확인함

## 2. **Preliminary**

- notation 정리
  - input sequence $$\mathit{\mathbf{s}}$$
  - input sequence length $$n$$
  - input token $$s_i(i\in [n])$$
  - $$d$$-dimensional embedding vector of a input token $$\mathit{\mathbf{v}}_i=\text{Embed}(s_i)\in \mathbb{R}^d$$
  - positional embedding $$\mathit{\mathbf{p}}_i \in \mathbb{R}^d$$
  - embedded input 은 다음과 같이 단축화되어 표현 가능: $$\mathbf{\mathit{X}}^{(0)}=[\mathit{\mathbf{v}}_1+\mathit{\mathbf{p}}_1, ..., \mathit{\mathbf{v}}_n+\mathit{\mathbf{p}}_n]^{\intercal}\in \mathbb{R}^{n\times d}$$
  - Number of transformer blocks $$L$$
  - Transformer layer depth $$L$$, width $$d$$, number of heads $$H$$
  - query, key, value, output parameter matrices of the $$h$$-th head$$\mathbf{\mathit{W}}_{Q}^{(l,h)}, \mathbf{\mathit{W}}_{K}^{(l,h)}, \mathbf{\mathit{W}}_{V}^{(l,h)}, \mathbf{\mathit{W}}_{O}^{(l,h)}$$
  - causal mask $$M \in \{-\infty, 0\}^{n\times n}$$, $$M_{i, j} = - \infty$$ iff $$i<j$$

1. 식 (1) 각 Transformer layer의 인풋에 대한 계산

   $$\mathbf{\mathit{X}}^{(l)} = \mathbf{\mathit{X}}^{(l-1)} + \text{Attn}^{(l)}( \mathbf{\mathit{X}}^{(l-1)}) + \text{FFN}^{(l)} (\mathbf{\mathit{X}}^{(l-1)} + \text{Attn}^{(l)}(\mathbf{\mathit{X}}^{(l-1)}))$$, $$l \in [L]$$

2. 식 (2) Transformer layer 내 multi-head self-attention layer 계산

   $$\text{Attn}^{(l)}(\mathbf{\mathit{X}}) = \Sigma^{H}_{h=1}\text{softmax}(\mathbf{\mathit{X}}\mathbf{\mathit{W}}_{Q}^{(l,h)} (\mathbf{\mathit{X}}\mathbf{\mathit{W}}_{K}^{(l,h)})^{\intercal}+M)\mathbf{\mathit{X}}\mathbf{\mathit{W}}_{V}^{(l,h)} W_O^{(l,h)}$$

3. 식 (3) Transformer layer 내 feed-forward network 계산

   $$\text{FFN}^{(l)} (\mathbf{\mathit{X}}) = \sigma(\mathbf{\mathit{X}}\mathbf{\mathit{W}}_1^{(l)})\mathbf{\mathit{W}}_2^{(l)}$$

## 3. **CoT is the Key to Solving Mathematical Problems**

- Arthimetic Expression과 Linear Equation을 해결하는 두 테스크에 대해 고민

<img width="346" alt="math" src="https://github.com/lhy0718/lhy0718.github.io/assets/11364584/1c87a254-d5c4-4629-beee-ed93f9fc56dd">

- 두 가지 수학 테스크의 분석을 단순화하기 위해 유한체 (finite field)로 일반화

### 유한체 (Finite field)

- 간단하게, $$p$$개의 숫자로 사칙연산이 정의되는 체계를 의미

> 💡 체론에서 유한체(有限體, 영어: finite field) 또는 갈루아 체(영어: Galois field)는 유한개의 원소를 가지는 체이다.
> 유한체는 항상 양의 표수 $$p$$를 갖는다 ($$p$$는 소수). 표수가 $$p$$인 유한체의 크기는 항상 $$p$$의 거듭제곱($$p^n$$, $$n$$은 자연수)이다. 크기가 $$p^n$$인 유한체는 $$\mathbb{F}_{p^n}$$이라고 쓴다.

출처: 위키백과

- 대수학에서 “체”(field)란 일반적인 사칙연산이 적용되는 대수 구조를 의미
  - 결합법칙, 분배법칙, 항등원, 역원 등의 속성이 만족됨
  - 유리수나 실수에서는 위 규칙들이 성립하지만, 이것들은 무한한 element들을 가지고 있음
- 유한체는 사칙연산의 결과에 나머지연산(modular)를 취하여 element를 유한한 $$p$$개로 제한
- 예를 들어 유한체 $$\mathbb{Z}_5$$를 가정하면, 2+3은 0이 됨 (2+3=5 를 5로 나눈 나머지가 0이므로)
  - 마찬가지로 유한체 $$\mathbb{Z}_5$$에서 2⨉3=1, 2-3=4, 2÷3=1 이 됨
- Section 3에서는 무한한 토큰을 방지하기 위해 유한체 $$\mathbb{Z}_p$$를 가정
  - 이는 evaluating arithmetic expression과 solving linear equation 테스크에서 모두 잘 정의됨
  - 논문에서는 각 테스크를 $$\text{Arithmetic}(n, p)$$ 과 $$\text{Equation}(m, p)$$으로 나타냄 ($$n, m$$은 최대 인풋 길이, $$p$$는 토큰의 수)

### 회로 복잡도 (**Circuit complexity)**

> 💡 이론 컴퓨터 과학에서 회로 복잡도는 부울 함수를 계산하는 부울 회로의 크기 또는 깊이에 따라 부울 함수를 분류하는 계산 복잡도 이론의 한 분야이다.
>
> 출처: 영문 위키피디아

회로 복잡도에 따르면 여러 복잡도 종류(Complexity class)가 있음 (아래 예시 말고도 더 있음)

- $$\textsf{TC}^0$$: TC Class의 첫번째 클래스로, 입력 개수에 제한이 없는 AND, OR, NOT, 그리고 majority gate으로 이루어진 상수 깊이, 다항 크기의 부울 회로로 결정되는 모든 언어를 포함
  - n개의 n-bit 숫자 소팅, 두 개의 n-bit 곱셈 등의 문제를 포함
- $$\textsf{AC}^0$$: AC 계층의 가장 작은 클래스. 입력 개수에 제한이 없는 AND, OR 게이트로 구성된 $$O(1)$$ 깊이 및 다항 크기의 회로 종류를 포함
  - 정수형 덧셈 뺄셈 등을 포함
- $$\textsf{NC}^i$$: 최대 2개의 입력과, 깊이 $$O((\text{log}~n)^i)$$의 다항 개수의 게이트를 갖는 균일 부울 회로로 결정할 수 있는 결정 문제의 클래스
  - 또는 다항 개수의 프로세서를 가진 병렬 컴퓨터에서 $$O((\text{log}~n)^i)$$ 시간 안에 풀 수 있는 결정 문제의 클래스
- $$\textsf{P}$$: 결정론적 튜링 기계로 다항 시간 안에 풀 수 있는 판정 문제를 모아 놓은 복잡도 클래스
- 복잡도 클래스의 포함관계는 다음과 같음

  > 💡 $$\textsf{NC}^0 \subsetneq \textsf{AC}^0 \subsetneq \textsf{TC}^0 \subset \textsf{NC}^1 \subset \textsf{P}$$

### **Theoretical results**

- DNN은 크기가 무한하면 이론적으로 universal function approximator 이므로, 모델의 크기가 충분히 크다면 모든 수학문제에 대해 한 번에 답을 맞출 수 있겠지만, 이는 모델이 특정 테스크를 해결하는데 필요한 모델 크기에 대한 정보를 주지는 않는다.
- 따라서 complexity theory의 관점에서 문제를 파악한다.
- **log-precision Transformer**
  - 인풋 크기가 최대 $$n$$일 때, 파라미터에 $$O(\text{log}~n)$$의 bit precision을 가지는 부동소수점을 저장할 수 있는 Transformer 모델을 의미
  - 예를 들어 32bit precision, 최대 인풋 길이 2048인 GPT모델
  - 일반적으로 bit precision이 최대 인풋 길이 보다 작음
  - log-precision은 각 뉴런이 input길이의 polynomial한 개수의 값을 가짐을 함의함
  - 아래 2개의 공리는 두 종류의 수학문제에 대해 굉장히 큰(prohibitively large) 네트워크가 필요함을 보여줌

> 📒 **Theorem 3.1**
>
> $$\textsf{TC}^0 \neq \textsf{NC}^1$$을 가정. 임의의 소수 $$p$$, 임의의 정수 $$L$$, 임의의 다항식 $$Q$$에 대해,
> Section 2에서 정의된 깊이 $$L$$, hidden dimension $$d \leq Q(n)$$인 log-precision autoregressive Transformer가 $$\text{Arithmetic}(n, p)$$ 을 해결할 수 없는 problem 크기 $$n$$이 존재

- 증명
  Appendix D.2, E.2 참조 ㅠㅠ

> 📒 **Theorem 3.2**
>
> $$\textsf{TC}^0 \neq \textsf{NC}^1$$을 가정. 임의의 소수 $$p$$, 임의의 정수 $$L$$, 임의의 다항식 $$Q$$에 대해,
> Section 2에서 정의된 깊이 $$L$$, hidden dimension $$d \leq Q(m)$$인 log-precision autoregressive Transformer가 $$\text{Equation}(m, p)$$2 을 해결할 수 없는 problem 크기 $$m$$이 존재

- Theorem 3.1과 3.2는 회로 복잡도 이론 (circuit complexity theory)으로 증명됨
- 다항 크기의 깊이가 제한된(bounded-depth) log-precision Transformers는 복잡도 상한이 $$\textsf{TC}^0$$인 앝은(shallow) 회로 클래스를 나타냄
- 반면 두 종류의 수학문제의 복잡도는 $$\textsf{NC}^1$$을 하한으로 함 ($$\textsf{NC}^1$$-완전 문제로 환원하여)
- **결과적으로, 두 복잡성 클래스가 붕괴되지 않는 한(즉, $$\textsf{TC}^0 = \textsf{NC}^1$$), Transformer 모델에 의해 수학문제를 해결하기 본질적으로 어렵다.**

### CoT의 경우에는 어떨까?

> 📒 **Theorem 3.3.**
> Fix any prime $$p$$. For any integer $$n > 0$$,
> there exists an autoregressive Transformer defined in Section 2 with constant hidden size $$d$$ (independent of $$n$$), depth $$L = 5$$, and 5 heads in each layer that can generate the CoT solution defined in Appendix B for all inputs in $$\text{Arithmetic}(n, p)$$.
> Moreover, all parameter values in the Transformer are bounded by $$O(\text{poly}(n))$$.

> 📒 **Theorem 3.4.**
> Fix any prime $$p$$. For any integer $$m > 0$$,
> there exists an autoregressive Transformer defined in Section 2 with constant hidden size $$d$$ (independent of $$m$$), depth $$L = 4$$, and 5 heads in each layer that can generate the CoT solution defined in Appendix B for all inputs in $$\text{Equation}(m, p)$$.
> Moreover, all parameter values in the Transformer are bounded by $$O(\text{poly}(m))$$.

> 📒 **Remark 3.5.**
> Theorem 3.3과 3.4의 매개 변수에 대한 다항식 상한은 이러한 Transformer가 정확성을 잃지 않고 log-precision을 사용하여 구현될 수 있음을 쉽게 의미함. 이것이 어떻게 달성될 수 있는지에 대한 자세한 논의는 Appendix A.3을 참조

#### Theorem 3.3, 3.4에 대한 증명 스케치

- softmax attention head 은 COPY와 MEAN 연산을 수행할 수 있음 (Lemma C.7, C.8)
  - 이 두 연산은 병렬 컴퓨팅에서 “gather/scatter” 연산으로 볼 수 있음
- 반면, Transformer layer의 FFN은 곱셈 (Lemma C.1), conditional selection (Lemma C.4), lookup table (Lemma C.5)와 같은 기본적인 연산을 각각의 자리에서 수행할 수 있음
  - 이러한 기본 연산을 "instruction"으로 하고 자동 회귀 생성을 루프로 취급함으로써, 상당히 복잡한 작업을 해결할 수 있는 "프로그램"을 작성할 수 있음
  - Appendix D.1과 E.1에 자세히 설명된 바와 같이, 두 수학 문제에 대해 CoT 시퀀스를 생성할 수 있는 병렬 알고리즘을 구축하여 증명함

### Section 3 에서 논의할 점들

1. Transformer의 고유한 능력
   - Transformer의 softmax attention, multi-head, FFN 등의 구성 요소의 중요성
   - 일정한 크기의 RNN은 동일한 CoT 형식을 사용하여 위의 수학 작업을 해결할 수 없음 (Appendix F.2)
2. CoT가 실제 인간의 사고와 닮아있음
   - CoT가 읽을 수 있는 수학 언어 형식으로 쓰여져 있으며, 인간이 해결책을 쓰는 방식과 크게 닮음
   - 넓은 의미에서, LLM이 문법적으로 정확한 문장을 통해 의미 있는 인간의 생각을 전달할 가능성이 있다는 것을 정당화
3. CoT + LLM이 Theorem 3.1, 3.2에서 설명된 불가능한 결과를 우회할 수 있는 이유는?
   - CoT를 사용함으로써, 생성된 출력이 반복적으로 입력
   - 출력 토큰 간의 의존성은 CoT 솔루션의 길이에 비례하는 깊이를 가진 (길이 $$L$$보다 긴) 훨씬 더 깊은 회로로 이어짐
   - 고정된 Transformer (혹은 회로) 내에서 재귀 절차가 반복되더라도, 표현력은 여전히 $$\mathsf{TC}^0$$을 훨씬 넘어설 수 있음
     - 충분한 수의 CoT 단계를 통해 autoregressive Transformers는 P-완전 문제를 해결할 수도 있음 (Section 4)

Section 4 부터는 다음에 이어서 설명하겠습니다…

## 코멘트

- ABSA에 CoT를 사용하는 것의 이점을 이론적으로 분석할 때의 접근법을 생각해봄
- 이 논문과 같이 ABSA task의 계산복잡도를 계산하여 … (그 다음에 어떻게 하지)
  - ABSA task를 단순한 문제로 환원 후 계산 복잡도를 계산
