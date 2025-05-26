---
title: "[논문리뷰] Embodied Agent Interface- Benchmarking LLMs for Embodied Decision Making (NeurIPS 2024)"
date: 2025-05-27 08:20:58 +0900
categories:
  - Paper Review
tags:
  - NeurIPS 2024
  - Decision Making
---

본 논문은 다양한 과제와 평가 지표를 통합한 EMBODIED AGENT INTERFACE를 제안하여, 대형 언어 모델(LLM)의 신체화된 의사결정 능력을 세부적으로 평가하고 장단점을 분석한다.

---

# 1 Introduction

- 대형 언어 모델(LLMs)은 인간의 지시(예: "냉장고 청소", "가구 광택")를 따라 다양한 디지털 및 물리적 환경에서 일련의 행동으로 목표를 달성하는 구현된 의사결정 에이전트를 구축하는 강력한 도구로 부상함 [1–3].
- 하지만 구현된 의사결정에서 LLM의 완전한 능력과 한계에 대한 이해는 제한적임.
- 기존 평가 방법은 다음 세 가지 주요 한계로 인해 포괄적 통찰을 제공하지 못함:
  1. 구현된 의사결정 작업의 표준화 부족
  2. LLM이 인터페이스하거나 구현 가능한 모듈의 표준화 부족
  3. 단일 성공률 외의 세밀한 평가 지표 부족

- 본 논문에서는 이 문제들을 해결하기 위해 **EMBODIED AGENT INTERFACE**를 제안함.

---

### 1) 목표 명세의 표준화

- 구현된 에이전트가 목표를 달성하기를 원하지만, 목표 명세 및 성공 평가 기준은 도메인과 유사 작업마다 크게 다름.
- 예를 들어, BEHAVIOR [4]는 상태 목표(예: "not_stained(fridge)") 달성에 중점 두는 반면, VirtualHome [5]는 행동에 시간적 순서 제약을 부여하여 시간 확장 목표를 사용함.
- EMBODIED AGENT INTERFACE는 객체 중심의 상태 및 행동 표현을 구현하며, 목표는 궤적 전반의 과제 성공 조건을 정의하는 **선형 시간 논리(LTL)** 식으로 기술함.
- LTL은 상태 기반 목표와 시간 확장 목표 모두를 지정할 수 있고 여러 대안적 목표 해석을 지원함.

---

### 2) 모듈 및 인터페이스의 표준화

- 기존의 LLM 기반 구현 에이전트 프레임워크는 추가 지식과 외부 모듈 가용성에 따라 서로 다른 가정을 함.
- 예:
  - Code as Policies [6], SayCan [2]: primitives 기반 행동 시퀀싱에 LLM 사용
  - LLM+P [7]: 목표 해석에 LLM 사용 및 PDDL 플래너와 연동
  - Ada [8]: 고수준 PDDL 생성 후 저수준 플래너 사용
- 이들간 입출력 명세가 달라 비교 및 평가가 어려움.
- EMBODIED AGENT INTERFACE는 객체 중심 및 LTL 기반 작업 명세 위에 다음 네 가지 핵심 능력 모듈을 형식화함 (그림 1 참고):
  - 목표 해석 (Goal Interpretation)
  - 하위 목표 분해 (Subgoal Decomposition)
  - 행동 시퀀싱 (Action Sequencing)
  - 전이 모델링 (Transition Modeling)
- 각 모듈의 입출력 명세를 정의하여 LLM이 환경 내 다른 모듈과 인터페이스할 수 있도록 함.
- 이는 다양한 LLM 및 외부 모듈 통합을 자동으로 가능하게 함.
- 예를 들어, 하위 목표 분해 모듈은 초기 상태와 과제 목표를 받아 하위 목표 궤적(예: 천을 적시는 것 → 잡는 것 → 냉장고 옆에 서는 것 → 냉장고 청소 완료)을 생성함 (표 1 참조).

---

### 3) 광범위한 세밀 평가 지표의 표준화

- 기존 LLM 기반 구현 의사결정 평가는 대체로 단일 작업의 성공률 위주로 단순화됨.
- 최근 LOTA-Bench [9]는 행동 시퀀스 생성 평가는 하지만 세밀한 계획 오류 분석은 지원하지 않음.
- EMBODIED AGENT INTERFACE는 객체 중심 및 분해된 상태, 행동 표현을 활용해 다음과 같은 다양한 세밀 평가 지표를 구현함:
  - 환각(hallucination) 오류, 행동 순서 오류, 객체 활용 오류 등 다양한 계획 오류 자동 탐지 가능
- 그림 3은 GPT-4o가 2개 시뮬레이터의 4개 능력 모듈에서 범한 다양한 오류 유형을 보여줌.
- 각 모듈별 평가는 다음 두 측면을 포함:
  - 궤적 평가: 생성된 계획이 시뮬레이터에서 실행 가능한지 판단
  - 목표 평가: 계획이 적절한 결과를 달성하는지 판단
- 목표 평가는 목표 해석, 행동 시퀀싱, 하위 목표 분해 모듈에 적용되고,
- 궤적 평가는 행동 시퀀싱, 하위 목표 분해, 전이 모델링 모듈에 적용됨.

---

### 주요 실험 및 결과

- EMBODIED AGENT INTERFACE를 BEHAVIOR [4]와 VirtualHome [5] 두 벤치마크에 구현, 18개 LLM 평가 진행.
- Figure 3은 BEHAVIOR에서 5개 대표 LLM 성능 시각화.
- 주요 결과:
  - 대부분의 LLM은 자연어 지시를 환경의 구체적 상태(객체, 객체 상태, 관계)로 충실히 번역하는 데 어려움을 겪음. 
    - 예: "drinking water" 작업에서 중간 하위 목표를 최종 목표로 잘못 예측(open(freezer))
  - 추론 능력 향상이 필수적임.
    - 궤적 실행 불가 오류 45.2%, 누락 단계 19.5%, 추가 단계 14.2%가 많음.
    - 주로 행동 전제조건 무시(예: 앉거나 누워 있음 상태를 고려하지 않고 일어서기 행동 누락, 닫힌 객체 열기 누락) 때문.
    - 이미 달성한 목표에 대한 불필요 행동 포함하는 추가 단계 오류도 빈번함.
  - 궤적 길이 증가 시 궤적 평가 성능 저하, 환경 복잡도 증가 시 목표 평가 성능 저하 관찰됨.
  - LLM 오류에 객체와 행동의 환각뿐 아니라 보고 편향(reporting bias)이 포함됨.
    - 예: "put the turkey on the table"은 실제로 "접시 위에 칠면조 올리고 접시를 테이블 위에 올리는 것"으로 해석되어야 함.
  - 하위 목표 분해 모듈은 계획 단순화를 위해 설계됐지만, 목표를 실행 가능한 단계로 선언적으로 쪼개야 해서 행동 시퀀싱만큼 복잡함.

---

### 기타 분석

- 모듈의 강건성에 대한 민감도 분석, 파이프라인 방식과 모듈화 방식 비교, 재계획(replanning) 등의 정량적 분석 수행.
- LLM과 외부 모듈 통합 가능성을 모색하는 데 목적.

---

### LLM 비교 및 성능 요약

- o1-preview 모델이 BEHAVIOR 시뮬레이터에서 특히 뛰어나 전체 74.9% 성능 달성 (64.2% 대비 우위).
- o1-preview는 VirtualHome에서 목표 해석, BEHAVIOR 및 VirtualHome에서 행동 시퀀싱, 전이 모델링, 하위 목표 분해에서 강점.
- Claude-3.5 Sonnet는 BEHAVIOR의 목표 해석과 VirtualHome의 전이 모델링에서 강세.
- Mistral Large는 VirtualHome에서 행동 시퀀싱 성능 우수.

---

# 2 Embodied Agent Interface

- **Embodied 의사결정 문제 표현**:  
  - 표현식: $$\langle U, S, A, g, \phi, \bar{a} \rangle$$  
  - 구성 요소:  
    - 객체 집합: $$o \in U$$  
    - 상태 집합: $$s \in S$$  
    - 행동 집합: $$a \in A$$  
    - 목표: $$g$$  
    - 하위 목표(서브골): $$\phi$$  
    - 행동 궤적: $$\bar{a}$$  
  - 설명: 객체 기반(Object-centric) 환경 추상화로, 객체, 상태, 행동, 목표, 하위 목표, 행동 경로를 언어적 표현으로 모델링함.

- **능력 모듈 네 가지 정의**:  
  - 표현식: $$\langle G, \Phi, Q, T \rangle$$  
  - 각 모듈 및 역할:  
    - 목표 해석 모듈 $$G$$ (Goal Interpretation Module)  
    - 하위 목표 분해 모듈 $$\Phi$$ (Subgoal Decomposition Module)  
    - 행동 순서화 모듈 $$Q$$ (Action Sequencing Module)  
    - 전이 모델링 모듈 $$T$$ (Transition Modeling Module)  
  - 목적: LLM(대형 언어 모델)이 이들 모듈과 표준화된 입출력 사양으로 상호작용 가능하도록 함.

- **객체 중심 모델링 초점**:  
  - 상태: 환경 내 엔티티 간 관계적 특징으로 표현  
  - 행동: 엔티티 이름을 입력으로 받는 함수로 정의, 환경에서 실행 가능  
  - 목표 및 하위 목표: 선형 시제 논리(LTL, Linear Temporal Logic) 수식을 활용해 상태와 행동 위에 정의됨  

- **논문의 주요 내용**:  
  본 논문에서는 위의 구성 요소와 모듈에 대해 상세하게 정의하고 구현함.

---

# 2.1 Representation for Objects, States and Actions

- 상태(state)는 튜플 $$s = \langle U, F \rangle$$로 표현됨
  - $$U$$: 객체들의 우주(universe)로, 고정된 유한 집합으로 가정함
  - $$F$$: 관계적 불리언 특징(relational Boolean features)들의 집합
- 각 특징 $$f \in F$$는 표(table) 형태이며, 각 항목은 객체들의 튜플 $$(o_1, \cdots, o_k)$$에 연관되어 있음
  - 여기서 $$k$$는 특징의 차수(arity)임
  - 각 항목 값은 해당 상태에서의 특징 값임
- 행동(actions)은 객체들을 입력으로 받는 원시 함수(primitive function)처럼 볼 수 있음
  - 행동은 $$\langle name, args \rangle$$로 표현됨
- 본 논문에서는 상태와 행동이 추상적 언어 형식으로 기술되는 작업에 초점
  - 객체 상태 예시: $$\text{is-open(cabinet1)}$$
  - 관계 예시: $$\text{is-on(rag0, window3)}$$
  - 행동 예시: $$\text{soak(rag0)}$$

---

# 2.2 Representation for Goals, Subgoals, Action Sequences, and State-Action Trajectories

- **목표, 하위목표, 행동 시퀀스의 표현**  
  - EMBODIED AGENT INTERFACE에서는 목표 $$ g $$, 하위목표 $$ \phi $$, 행동 시퀀스 $$ \bar{a} $$를 선형 시계 논리(Linear Temporal Logic, LTL) 공식으로 모델링함.
  - 그 이유는 다음 두 가지 중요한 요구사항 때문:
    1. 상태 기반 목표 및 시간 확장 목표를 표현할 수 있는 표현력 있고 간결한 언어 필요.
    2. 다양한 LLM 기반 모듈 간 통합된 인터페이스 필요.  
  - LTL은 이 두 가지 요구사항을 모두 충족시킴.

- **LTL 공식의 특징**  
  - 상태 제약(예: 하위목표 달성 필요), 행동 제약(예: 특정 행동 수행 필요), 그리고 시간적 순서(예: 모든 설거지 후 요리)를 표현 가능.
  - 시간 연결자("eventually")와 명제 논리 연결자("or")를 결합해 대체 목표나 궤적도 유연하게 묘사할 수 있음.
  - 단일 서술 언어 사용 덕분에 정확도를 측정하는 통합 메트릭 설계가 가능함 (세부사항은 Appendix C.1 참조).

- **사용하는 LTL의 구체적 특성**  
  - 전체 LTL 공식 중 유한 궤적에 적용 가능한 단편(fragment) 사용.
  - 두 종류의 원자 명제(atomic propositions) 존재:
    - 상태 명제 (객체 속성 및 관계)
    - 행동 명제
  - 포함 논리 연산자:  
    - 불리언 연결자: $$\wedge$$, $$\vee$$, $$\neg$$, $$\Rightarrow$$
    - 1차 논리 수량자: $$\forall$$, $$\exists$$, $$\exists= n$$ (정확히 $$ n $$개 객체가 조건 만족)
    - 시간 연결자: then

- **LTL 공식의 의미론적 해석**  
  - LTL 공식은 궤적 분류자(trajectory classifier) 역할:  
    - 함수 $$\text{eval}(\phi, \bar{t})$$는 상태-행동 시퀀스 $$\bar{t}$$에 대해 공식 $$\phi$$를 평가.
    - $$\text{eval}(\phi, \bar{t}) = \text{true}$$이면 이 시퀀스가 $$ \phi $$를 만족한다고 함.
  - 상태 공식 $$\phi$$ (then 미포함)일 경우:  
    $$
    \text{eval}(\phi, \bar{t}) = \exists t. \; \phi(s_t)
    $$  
    ("언젠가" 목표가 만족됨)
  - then 연결 공식 $$\phi_1 \text{ then } \phi_2$$의 경우:  
    $$
    \text{eval}(\phi_1 \text{ then } \phi_2, \bar{t}) = \exists k. \; \phi_1(\bar{t}_{\leq k}) \wedge \phi_2(\bar{t}_{> k})
    $$  
    ($$\phi_2$$가 $$\phi_1$$ 뒤에 달성됨)  
    여기서 $$\bar{t}_{\leq k}$$와 $$\bar{t}_{> k}$$는 각각 시퀀스의 접두부와 접미부를 의미.

- **현재 상태 및 확장 가능성**  
  - 현재는 "globally", "until" 같은 다른 시간 연결자는 구현되지 않았음.
  - 전체 프레임워크는 향후 확장 가능.

- **예시**  
  - "browse Internet" 작업을 위한 하위목표 계획 LTL 예시:  
    $$
    \text{ontop}(\text{character}, \text{chair}) \text{ then } \text{holds\_rh}(\text{character}, \text{mouse}) \wedge \text{holds\_lh}(\text{character}, \text{keyboard}) \text{ then } \text{facing}(\text{character}, \text{computer})
    $$

- **추가 정보**  
  - LTL 공식에 대한 상세 내용은 Appendix C.3에 수록.

---

# 2.3 Ability Module 1: Goal Interpretation

- **입력-출력 명세**  
  - 목표 해석 모듈은 초기 상태 $$s_0$$와 자연어 지시 $$l_g$$를 입력으로 받음  
  - 출력으로는 상징적 계획자가 입력으로 사용할 수 있는 형식적 목표 명세인 LTL 목표 $$\hat{g}$$ 생성  
  - 본 논문에서는 순서가 지정된 행동 시퀀스와 최종 상태에서 만족해야 하는 명제들의 합성(conjunction)으로 구성된 단순한 LTL 목표만 생성  

- **평가 기준**  
  - LTL 목표는 생성된 $$\hat{g}$$와 참 목표 $$g$$를 직접 비교하여 평가 가능  
  - 생성된 $$\hat{g}$$는 단순 LTL 목표로 제한하였으나, 참 목표 $$g$$는 단순할 필요 없음  
  - 따라서, 객체 집합 $$U$$를 입력으로 받아 참 목표 $$g$$를 포함하는 단순 LTL 목표들의 집합 $$g_0, g_1, \ldots, g_k$$로 변환하는 함수 $$G$$ 정의 (세부 구현은 부록에 설명)  

- **F1 스코어 계산 방식**  
  - 임의의 단순 LTL 목표 $$g = a_1 \text{ then } \ldots a_k \text{ then } (p_1 \wedge \ldots \wedge p_\ell)$$에 대해,  
  - 동작 시퀀스 $$\{a_i\}_{i=1}^k$$를 하나의 요소로 보고,  
  - 집합 표현:  
    $$
    \text{set}(g) = \{\{a_i\}_{i=1}^k\} \cup \{p_i\}_{i=1}^\ell
    $$  
  - 생성 목표 $$\hat{g}$$에 대해,  
  - $$g$$의 변환 집합 내 모든 $$g_i$$와 $$\hat{g}$$ 사이의 F1 집합 매칭 점수를 계산하여 최고값을 최종 F1 점수로 정의:  
  $$
  F1(g, \hat{g}) = \max_{g_i \in G(g, U)} F1(\text{set}(g_i), \text{set}(\hat{g}))
  $$

---

# 2.4 Ability Module 2: Subgoal Decomposition

- **입력-출력 사양 (Input-Output Specification)**  
  - 하위 목표 분해 모듈은 작업 ⟨s0, g⟩를 입력으로 받고, 하위 목표들의 시퀀스 ¯ϕ = \{ϕ_i\}^k_{i=1}를 생성한다.  
  - 각 ϕ_i는 LTL(Linear Temporal Logic) 공식이다.  
  - 전체 시퀀스 ¯ϕ 또한 단일 LTL 공식으로 표현될 수 있다.  
  - 하위 목표 분할에 관한 선택 과정은 부록 D.3을 참고할 수 있다.

- **평가 지표 (Evaluation Metric)**  
  - 하위 목표 분해 모듈의 성능 평가는 커스터마이즈된 플래너를 사용하여 이를 동작 시퀀스 ¯a로 세분화하여 수행한다.  
  - 하위 목표-행동 매핑 함수 $$AM(\bar{\phi}, s_0)$$는 ¯ϕ의 LTL 표현과 초기 상태 s_0를 입력으로 받아 상태-행동 시퀀스 ¯t를 생성한다.  
  - 이 작업은 너비 우선 탐색(breadth-first search) 기법으로 구현된다.  
  - 평가에는 동작 시퀀싱에서 사용하는 동일한 지표인 경로 실현 가능성(trajectory feasibility)과 목표 만족(goal satisfaction)을 사용한다.

- **추가 제약 및 평가 방법**  
  - 각 ϕ는 여러 다른 동작 시퀀스로 구체화될 수 있기 때문에, 하위 목표마다 가능한 동작 수를 제한하여 유한한 동작 시퀀스 집합 ¯a_i를 생성한다.  
  - 각 ¯a_i에 대해 경로 실현 가능성 및 목표 만족 지표를 계산한다.  
  - 최종적으로 모든 ¯a_i 중 최대 점수를 해당 ϕ의 경로 실현 가능성과 목표 만족 점수로 보고한다.

---

# 2.5 Ability Module 3: Action Sequencing

- **입력-출력 명세**
  - 액션 시퀀싱 모듈은 작업 ⟨s0, g⟩과 전이 모델 M을 입력으로 받음.
  - 출력은 액션 시퀀스 $$\bar{a} = \{a_i\}_{i=1}^n$$를 생성함.

- **평가 지표**
  1. **경로 실행 가능성 평가** (Trajectory Feasibility Evaluation)
     - 경로가 실행 가능한지, 즉 모든 액션이 실현 가능한지를 평가함.
     - 시뮬레이터에서 초기 상태 $$s_0$$부터 시퀀스 $$\bar{a}$$를 실행.
     - 실행 불가능한 액션이 나타나면 수행이 조기 종료될 수 있음.
     - 실패 원인은 다음과 같이 분류됨:
       - 누락된 단계 (missing steps)
       - 추가된 단계 (additional steps)
       - 잘못된 시간 순서 (wrong temporal order)
       - 어포던스 오류 (affordance errors)

  2. **목표 만족 평가** (Goal Satisfaction Evaluation)
     - $$\bar{a}$$ 실행 후 목표 $$g$$가 만족되었는지를 평가함.
     - 실행 결과 $$T = \langle \{s_i\}_{i=0}^m, \{a_i\}_{i=1}^m \rangle$$ 획득.
     - 함수 $$\text{eval}(g, T)$$을 이용해 목표 만족 여부 점검.
  
  3. **부분 목표 만족 평가** (Partial Goal Satisfaction Evaluation)
     - 목표 $$g$$ 내의 “하위 목표(subgoals)” 중 $$\bar{a}$$에서 만족된 비율을 평가.
     - $$g$$에서 파생된 간단한 LTL 목표 $$g_i$$들을 고려함.
     - LTL 목표 형태:
       $$
       g_i = a_1 \; \text{then} \ldots \text{then} \; a_k \; \text{then} \; (p_1 \wedge \ldots \wedge p_\ell)
       $$
     - $$\bar{a}$$ 내에 시퀀스 $$\{a_j\}_{j=1}^k$$와 동일한 부분 수열이 있으면 성공으로 간주.
     - 이후 최종 상태 $$s_m$$에서 모든 상태 명제 $$p_j$$를 평가하여 부분 점수를 부여.
     - 최종 부분 성공률은 다음과 같이 정의:
       $$
       \text{PartialSucc}(\bar{a}, g) = \max_{g_i \in G(g, U)} \text{PartialSucc}(\bar{a}, g_i)
       $$

---

# 2.6 Ability Module 4: Transition Modeling

- **입출력 명세**  
  - 전이 모델링 모듈은 작업 ⟨s₀, g⟩와 연산자 정의 집합 {oᵢ}를 입력으로 받아, 각 oᵢ에 대해 PDDL 연산자 정의를 생성함  
  - 본 모듈의 목표는 작업 해결을 위한 계획 생성에 필요한 행동의 형식적 정의를 만드는 것  
  - 평가 시, 실제 행동 궤적 ¯a에 기반해 관련 연산자 정의 {oᵢ}를 추출 (추가 세부사항은 부록 C.3 참조)  
  - 이후 LLM이 모든 연산자 {oᵢ}에 대해 전제조건과 효과 {⟨preᵢ, effᵢ⟩}를 생성함  

- **평가 지표**  
  1. **논리 일치 점수 (Logic Matching Score)**  
     - 생성된 preᵢ, effᵢ를 인간 전문가가 주석 처리한 실제 연산자 정의(preᵢ^gt, effᵢ^gt)와 비교  
     - 표면 형태 매칭 점수를 사용해 논리식 간 F1 기반 점수 산출  
     - 직관적으로, preᵢ와 preᵢ^gt가 명제들의 합성(conjunction)일 경우, 아래와 같이 명제 집합 간 매칭 점수로 계산  
       $$ F_1 = \text{set\_matching\_score}(\{propositions\_{pre_i}\}, \{propositions\_{pre_i^{gt}}\}) $$  
     - 더 복잡한 논리식(예: $$\forall x.\varphi(x)$$)은 재귀적으로 평가 (자세한 내용은 부록 C.3 참조)  
     - 효과(effects) 평가도 같은 방식으로 진행됨  

  2. **계획 성공률 (Planning Success Rate)**  
     - 여러 연산자의 전제조건과 효과가 실제로 실행 가능한 계획을 생성할 수 있는지 여부를 평가  
     - 외부 PDDL 플래너를 사용하여 초기 상태 s₀에서 목표 g를 달성하는 계획 실행 시도  
     - 단순화를 위해 g 내 목표만 사용하며, 행동의 하위 목표는 무시  
     - 플래너가 계획을 찾으면 성공률은 1, 그렇지 않으면 0  

- **요약: 평가 방법**  
  - 논리적 일치도 점수로 행동 정의의 정확성 측정  
  - 플래닝 성공률로 실제 계획 실행 가능성 평가

---

# 3 Dataset Annotations and Benchmark Implementations

- **선정 시뮬레이터**  
  - 복잡한 장기 과업(long-horizon tasks)에 초점을 맞춰 BEHAVIOR (B)와 VirtualHome (V)를 평가 시뮬레이터로 선정  
  - 시뮬레이터별 비교 및 선택 고려사항은 부록 M.1에 수록

- **주요 어노테이션**  
  - 목표(goal) 및 궤적(trajectory) 어노테이션 외에, 후속 효과가 없는 필요한 행동을 반영하는 Goal Action 어노테이션 도입  
  - 예: "고양이를 쓰다듬기(pet the cat)" 과업 내의 접촉 행동(touch)  
  - VirtualHome 작업 중 80.7%가 10단계 이상의 행동을 포함하며, 33%는 단계 길이가 10단계 이상임

- **표 2: 시뮬레이터 데이터셋 통계 요약**  
  | 항목               | VirtualHome | BEHAVIOR |
  |--------------------|-------------|----------|
  | 작업(task) 수       | 26          | 100      |
  | 작업 지침(instruction) 수 | 338         | 100      |
  | 목표(goal) 수          | 801         | 673      |
  | 상태(state) 수          | 340         | 153      |
  | 관계(relation) 수       | 299         | 520      |
  | 행동(action) 수         | 162         | -        |
  | 궤적(trajectory) 수      | 338         | 100      |
  | 단계(step) 수           | 2960        | 1460     |
  | 평균 단계(avg. step)    | 8.76        | 14.6     |
  | 전이 모델(transition model) 수 | 33          | 30       |
  | 전제조건(precondition) 수  | 99          | 84       |
  | 효과(effect) 수          | 57          | 51       |

- **BEHAVIOR 시뮬레이터 특징**  
  - 복잡한 과업 처리 가능  
  - BDDL 목표는 정량자(quantifiers)를 포함할 수 있음  
    - 예:  
      $$ (forpairs\ (?jar\ ?apple)\ (inside\ ?apple\ ?jar)) $$  
  - 정량자를 원자 명제(atomic propositions)만 포함하는 실체화된 목표로 변환 필요  
    - 예:  
      $$ and\ ((inside\ apple\_1\ jar\_1)\ (inside\ apple\_2\ jar\_2)) $$
  - 동일한 BDDL 목표에 대해 여러 가지 실체화된 목표(goal options)가 존재  
    - 예:  
      $$ ((inside\ apple\_2\ jar\_1)\ (inside\ apple\_1\ jar\_2)) $$
  - 평균 과업당 실체화된 목표 수: 6.7  
  - 평균 과업당 목표 옵션(goal options) 수: 4,164.4  
  - 데이터 분포 및 추가 통계는 부록 M.2에 상세 수록

- **시뮬레이터 구현**  
  - BEHAVIOR는 자체적으로 행동 전이 모델(action transition model) 레이어 미보유  
  - 이를 위해 행동 전이 모델을 포함하는 심볼릭 심레이터 EvalGibson 구현  
    - 에이전트는 30가지 행동으로 객체 상태 변경 가능  
    - 구현 세부사항은 부록 N.1에 수록
  - VirtualHome 심레이터도 정확한 평가 지원을 위해 수정됨(부록 N.2 참고)
  - 대규모 모델별 평가 설정은 부록 O에 상세 기술

---

# 4 Results

- **평가 개요**  
  - 18개의 오픈 가중치 및 독점 LLMs을 두 가지 시뮬레이터(BEHA VIOR, VirtualHome)의 네 가지 능력 모듈에서 평가.  
  - Table 3은 전반적인 결과 요약, Tables 4-7은 대표 LLM 4종의 모듈별 분석 결과 제공.  
  - Figure 5는 다양한 에러 유형 사례 제시.

- **모델 비교 (Figure 3)**  
  - 최고 성능 모델: o1-preview, Claude3.5 Sonnet, Gemini 1.5 Pro.  
  - o1-preview는 객체 상태(object states)를 제외한 모든 영역에서 선두, Gemini 1.5 Pro는 객체 상태 추론 능력에서 최고.  
  - 오픈 가중치 모델 중 최고는 Llama-3-70B, Mistral-Large-2402이나, 상용 모델 대비 성능 차 존재.

- **능력 모듈별 비교**  
  - o1-preview: BEHA VIOR에서 74.9%의 우위 달성 (VirtualHome 64.2% 대비).  
  - VirtualHome 목표 해석(goal interpretation), BEHA VIOR의 행동 순서(action sequencing), 상태 전이 모델링(transition modeling) 및 두 시뮬레이터 전반의 하위 목표 분해(subgoal decomposition)에서 선두.  
  - Claude-3.5 Sonnet: BEHA VIOR 목표 해석과 VirtualHome 상태 전이에 강함.  
  - Mistral Large: VirtualHome 행동 순서 우수.  
  - Mixtral-8x22B: 오픈 가중치 중 상태 전이에 강점, Llama-3 70B instruct는 목표 해석에서 우수.  
  - BEHA VIOR는 과제 시간이 길고(평균 14.6단계), VirtualHome은 상태 공간이 더 큼 → 궤적 실현 가능도 점수는 BEHA VIOR가 낮고 목표 해석 점수는 높음 → 궤적 평가와 시퀀스 길이, 목표 평가와 환경 복잡도는 반비례 관계.  
  - 목표 성공률(cofactors): 작업 목표 수, 행동 길이, 작업 객체 길이 등이 영향(상세는 Appendix E.5).

- **객체 상태 vs 관계 (relation)**  
  - 관계 목표(goal) 추론이 객체 상태 추론보다 어려움.  
  - 공간관계(spatial relation)의 재현율과 목표 만족률이 낮음 (Table 4, 5).  
  - 비공간 관계(예: hold)는 공간 관계보다 더 예측하기 어려움 (Table 6). 예: 칫솔 잡기(holding(toothbrush))는 양치 행위의 전제 조건.

- **주요 표 요약(일부)**
  - Table 3: 모델별 목표 해석, 행동 순서, 하위 목표 분해, 상태 전이 평가 및 평균 성능  
  - Table 4: 목표 해석의 논리형식 정확도(F1)  
  - Table 5: 행동 순서 및 하위 목표 분해의 목표 만족률(%)  
  - Table 6: 궤적 평가 결과 및 오류 유형 (문법, 런타임, 구문 분석 오류 등)  
  - Table 7: 상태 전이 모델링의 논리형식 정확도 및 플래너 성공률

- **오류 유형 (Figure 5)**  
  - 런타임 오류가 대부분이며, 문법 오류는 적음.  
  - 누락 단계(missing-step), 추가 단계(additional-step) 오류가 빈번.  
  - 잘못된 순서 오류, 사전조건 미충족 오류도 발생.  
  - LLM 출력 오류: 상태, 관계, 행동 목표를 놓치거나 환각(hallucination) 발생.  

- **능력 모듈별 구체 분석**  
  - **목표 해석 (Goal Interpretation)**  
    - LLM은 중간 하위 목표와 최종 목표를 구분하는 데 어려움. 예: GPT-4o는 중간 상태(open(freezer), inside(water, glass))를 최종 목표로 오인.  
    - 자연어 목표를 단어 단위로 기호(Symbolic) 변환하는 경향이 있으며 환경 상태에 근거하지 않음. (Appendix E.1 참고)  
  - **하위 목표 분해 및 행동 순서의 궤적 실현 가능성**  
    - 주요 오류는 런타임 오류, 추가 단계를 예측하는 경우가 많음.  
    - 예: 목표가 달성되었음에도 추가 행동을 실행하거나(예, 박스를 두 번 여는 경우) 선결 조건 미충족 상태에서 행동 시도.  
    - 행위 실행 순서 오류는 비교적 적음. (Appendix E.3)  
  - **하위 목표 분해 및 행동 순서의 목표 만족률**  
    - 객체 목표(toggled_on 등)가 관계 목표(ontop(agent, chair))보다 달성하기 쉬움. (Appendix E.2)  
  - **상태 전이 모델링 (Transition Modeling)**  
    - 객체 상태와 관계 유형별로 다섯 가지 능력 카테고리 평가(Appendix F.3).  
    - 관계 기반 전제 조건과 효과가 객체 상태보다 예측 난이도 높음.  
    - 예: Claude-3 Opus는 객체 상태(task)에서 63% 우수, 공간 관계에서는 약함.  
    - 객체 방향성(예: 시청을 위해 TV를 바라봐야 하는 상태) 추론 성능은 낮음.  
    - LLM은 효과(effect)에는 객체 상태를 과대 표현하고, 전제 조건(precondition)에는 과소 표현하는 경향 있음. 반면 공간 관계는 전제 조건엔 과대, 효과에는 과소 표현.  
    - 이로 인해 다운스트림 플래너가 계획을 생성해도 실제 환경에서는 실행 불가능한 경우 발생. (Appendix E.4, F 참고)

- **구체적인 과제 달성 오류 예:**
  - “serve a meal” 예시에서 “chicken”을 “table” 위에 놓는 목표를 잘못 예측(“plate” 위가 아님) → 자연어 표현의 편향 때문.  
  - “cleaning sneakers” 작업에서 onfloor 관계가 누락 → 대화 모델이 암묵적 공간 관계를 무시함.  
  - 정밀한 물리 관계는 구현에 필수적임.

- **로봇 에이전트 시스템 설계에의 시사점**  
  - LLM 기반 능력 모듈들의 통합 및 민감도 분석(Appendix F), 모듈화 대 파이프라인 기반 실험(Appendix G), 재계획(replanning) 실험(Appendix H) 수행.  
  - 에러 축적에도 불구하고 궤적 실현 가능도는 유사하게 유지되어 모듈 조합의 가능성 확인.  
  - 임베디드 의사결정에 대한 다양한 프롬프트 전략 비교 및 베스트 프랙티스 요약함(Appendix I).

---

# 5 Related Work

- 최근의 구현된 의사결정(embodied decision making) 연구들은 LLMs를 활용하여 다양한 작업을 수행하고 있음.  
- 부록 P와 표 8에서 관련 연구에 대한 종합적인 요약을 제공하며, 표 8은 빠른 개요를 위한 참고 자료임.  

- LLMs는 다음과 같은 다양한 모듈을 한 번에 결합하는 데 활용됨:  
  - chain-of-thought 프롬프트를 이용한 목표 해석(goal interpretation)과 행동 순서(action sequencing) 결합 [13–32]  
  - 목표 해석과 하위 목표 분해(subgoal decomposition) 결합 [2, 27, 33]  
  - 행동 순서와 하위 목표 분해 결합 [27, 34, 18, 35]  
  - 행동 순서와 전이 모델링(transition modeling) 결합 [8, 28, 32, 36, 37, 13, 38]  

- 본 연구의 목적은 LLMs와 다양한 의사결정 모듈 간 인터페이스를 표준화하여:  
  - 매끄러운 통합(seamless integration),  
  - 모듈 단위 평가(modular evaluation),  
  - 세밀한 지표(fine-grained metrics)를 지원하는 것임.  
- 이를 통해 LLM 활용의 효과적이고 선택적인 사용에 대한 시사점을 제공하고자 함.  

- 추가 관련 연구는 에이전트 인터페이스(agent interfaces) [39–43, 18, 44, 42, 45]와 시뮬레이션 벤치마크(simulation benchmarks)에 관한 내용이 부록 P에 수록되어 있음.  

## 표 8: LLMs를 활용한 구현된 에이전트 관련 선행 연구 분류

| 작업 | 관련 연구 |
| ------ | --------- |
| 목표 해석(Goal Interpretation) | [2, 7, 46, 47, 21, 48, 22–25, 27–32, 13–15, 49–53] |
| 하위 목표 분해(Subgoal Decomposition) | [2, 27, 34, 18, 33, 35, 54, 55] |
| 행동 순서(Action Sequencing) | [6, 8, 35, 56–59, 16, 43, 60, 17, 19, 61, 20, 42, 62, 14, 63–66, 3, 15, 45, 67–69] |
| 전이 모델링(Transition Modeling) | [8, 28, 32, 36, 70, 37, 13, 38] |

---

# 6 Conclusions and Future Work

- 본 논문에서는 체화된 의사결정을 위한 대형언어모델(LLM) 평가를 체계적으로 수행할 수 있는 평가 프레임워크 *EMBODIED AGENT INTERFACE*를 제안함.
- 주요 특징:
  1. LTL (Linear Temporal Logic) 수식을 이용한 목표 명세의 표준화
  2. 표준 인터페이스와 4가지 기본 능력 모듈을 통한 의사결정 과제의 통합
  3. 세밀한 평가 지표 제공 및 자동 오류 식별 기능 포함
- 현재 LLM의 한계점 및 오류 유형 분석:
  - 복잡한 목표 해석의 어려움
  - 추론 과정에서 발생하는 다양한 오류
  - 오류 발생에 기여하는 요인으로 이동 경로 길이, 목표 복잡성, 공간 관계 목표 등이 있음
- 한계점 및 향후 과제:
  - 현재 평가는 추상적 언어로 기술 가능한 상태, 행동, 목표에 제한되고, 환경 입력은 객체 간 관계 그래프로 추상화됨
  - 향후 연구에서는 감각 입력과 작동 출력까지 확장해야 하며, 이를 위해 비전-언어 모델(VLM) 등으로 모델 범위를 확장할 필요가 있음 (자세한 내용은 부록 K 참조)
  - 메모리 시스템(에피소드 메모리, 상태 메모리), 기하학적 추론, 네비게이션 통합 등도 확장 과제로 제시됨
