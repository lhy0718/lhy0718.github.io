---
title: "[논문리뷰] Inference and Learning in Dynamic Decision Networks Using Knowledge Compilation (AAAI 2024)"
date: 2025-05-26 16:20:48 +0900
categories:
  - Paper Review
tags:
  - AAAI 2024
  - Decision Making
---

본 논문은 동적 환경에서 불확실성을 고려한 의사결정 문제를 다루며, 동적 결정 네트워크(DDN)에 대해 벨만 업데이트를 지식 컴파일하여 미분 가능한 동적 결정 회로를 생성하고 이를 통해 기울기 기반 학습을 수행하는 새로운 알고리즘 mapl-cirup을 제안한다.

---

# 1 Introduction

- 베이지안 네트워크(Bayesian networks, BNs)는 불확실성 하의 현실 세계 과정을 모델링하는 데 널리 사용됨 (Koller and Friedman 2009; Russell and Norvig 2020).
- 하지만 BN에서의 추론(inference)은 계산적으로 매우 어려운 문제로, #P-hard에 해당함 (Roth 1996).
- 이 어려움을 해결하기 위해 최첨단 기법들은 지식 컴파일(knowledge compilation, KC)을 적용하는데, 조건부 독립성 및 지역 구조(local structures)를 활용함 (Chavira and Darwiche 2008).
- 최근에는 동적 모델(dynamic models)에 대해서도 KC가 성공적으로 적용되어, 지역 구조뿐만 아니라 시간에 따른 반복 구조도 활용함 (Vlasselaer et al. 2016).
- 동적 의사결정 네트워크(Dynamic decision networks, DDNs)는 동적 베이지안 네트워크(Dean and Kanazawa 1989; Murphy 2002)와 의사결정 이론 베이지안 네트워크(Howard and Matheson 1984; Bhattacharjya and Shachter 2007)를 결합하여 하나의 모델링 언어로 만듦 (Kanazawa and Dean 1989; Russell and Norvig 2020).
  - 이를 통해 마코프 의사결정 과정(MDPs, Markov decision processes) (Bellman 1957) 같은 문제를 표현할 수 있음.

- 예시 1 (Monkey DDN, Figure 1):
  - 원숭이가 의심스러운 진흙(H)으로 당신을 때리려 시도함.
  - 당신은 움직일지(M) 말지 결정 가능.
  - 만약 맞으면, 냄새가 날 수 있음(B).
  - 원숭이는 축하하며 다음 번에 때릴 확률이 줄어듦.
  - 결정과 현재 상태가 다음 상태(primed 변수들)에 영향을 줌.
  - 놓치면 원숭이가 더 화를 내고 집중도가 올라가 다음에 맞을 확률이 증가.
  - 움직이는 것은 맞을 확률을 낮춤.
  - 상태와 결정에 따른 보상(R)이 있음.

- Derkinderen과 De Raedt (2020)는 단일 단계 최대화 문제를 KC로 모델링하고 해결함.
  - 이 때 대수적 모델 카운팅(algebraic model counting) 프레임워크(Kimmig, Van den Broeck, and De Raedt 2017)를 활용하여, 한 번의 컴파일로 다양한 작업을 같은 컴파일된 도표 위에서 해결 가능.

- 본 연구에서는 대수적 모델 카운팅 프레임워크를 동적 환경에 적용하여 DDN을 해결하는 방법을 탐구함.
  - 또한 같은 컴파일된 도표를 매개변수 학습(parameter learning)에도 활용함.

- 주요 기여점:
  1. **mapl-cirup (Markov planning with circuit updates)** 알고리즘 제안 (발음은 ‘maple syrup’).
     - 고전적 가치 반복 알고리즘(value iteration)의 변형임.
     - KC를 이용하여 변수 간 의존성과 반복적인 시간 구조를 활용함.
  2. Chavira와 Darwiche (2008)의 아이디어를 확장하여 BN을 가중 논리식(weighted logic formulas)으로 인코딩하고, 효율적 추론을 위해 산술 회로(arithmetic circuits)로 지식 컴파일함.
  3. **동적 의사결정 회로(Dynamic decision circuits, DDCs)**를 새로 도입하여, DDN을 위한 컴파일 타겟으로 삼음.
     - 이를 통해 계획 문제(planning)를 DDC 내 확률 추론 문제로 환원함.
  4. 결과적 표현이 미분 가능하므로, MDP의 그래디언트 기반 매개변수 학습에도 활용 가능함.

---

# 2.1 Model Counting and Knowledge Compilation

- 이산 확률 변수로 확률 추론(probabilistic inference)을 수행하는 대표적인 기법 중 하나는 이를 가중 모델 카운팅(Weighted Model Counting, WMC)으로 환원하는 것임 (Darwiche 2009).
  - 특정 확률 추론 문제를 가중치가 부여된 명제 논리 공식으로 인코딩.
  - 공식의 가중치 계산은 목표 확률 계산과 동일해짐.
  - 예를 들어, Chavira와 Darwiche (2008)는 이 기법을 Bayesian Network(BN)의 추론에 사용.
  - 본 논문도 Boolean 변수에만 집중하며, 다중값 변수를 Boolean 변수 조합으로 인코딩함.

- **예제 2 (Weighted Model Counting):**
  - 이론 $$T = C \leftrightarrow A \lor B$$에 대해, 가중치 함수  
    $$w : \{a \mapsto 0.1, \neg a \mapsto 1 - 0.1, b \mapsto 0.2, \neg b \mapsto 1 - 0.2, c \mapsto 0.3, \neg c \mapsto 1 - 0.3\}$$  
  - 이론의 가중 모델 카운트 계산:  
    $$
    \begin{aligned}
    WMC(T, w) &= w(a)w(b)w(c) + w(a)w(\neg b)w(c) + w(\neg a)w(b)w(c) + w(\neg a)w(\neg b)w(\neg c) \\
    &= 0.006 + 0.024 + 0.054 + 0.504 = 0.588
    \end{aligned}
    $$
  - 여기서 0.588은 해당 이론이 만족될 확률.

- 문제점: 명제 논리 공식에 대한 WMC는 계산적으로 매우 어려우며, 이 문제는 정확히는 #P-난해함(Valiant 1979).
  - 예제 2 처럼 모든 만족 가능한 상태를 명시적으로 나열하는 것은 직관적으로 어려움.
  - 현실적으로 모든 상태를 일일이 나열하는 것은 불가능.

- 해결책: 지식 컴파일(Knowledge Compilation, KC) (Darwiche 2002) 기법 사용.
  - 핵심 아이디어: 명제 논리 공식을 WMC 연산 및 이에 따른 확률 추론이 다항 시간 내에 가능하도록 변환된 표현으로 컴파일함.
  - 중요한 장점: 컴파일은 가중치에 독립적임(weight agnostic).
    - 즉, 동일한 논리 공식을 공유하는 여러 WMC 문제를 단일 컴파일된 표현으로 해결 가능.
    - 이로 인해 계산 비용이 높은 컴파일 과정(오프라인)을 여러 질의 단계(온라인)에서 분산할 수 있어 반복 추론에 유리.

- **예제 3 (Knowledge Compilation):**
  - 예제 2의 논리 이론은 모든 가능한 모델을 명백히 나열한 방향 비순환 그래프(DAG)로 표현 가능 (그림 2a).
  - 그러나 KC를 사용하면 동일한 이론을 더 압축된 형태로 표현 가능 (그림 2b).
  - 이를 통해 효율적인 WMC 계산이 가능.
  - 논리 연결자 ∧, ∨는 각각 곱셈과 덧셈 연산으로 대체하고, 리터럴은 대응하는 가중치로 치환하면 산술 회로(arithmetic circuit)를 얻음 (그림 2c).
  - 이 회로를 평가하면 가중 모델 카운트를 계산할 수 있음.

---

# 2.2 Algebraic Model Counting

- WMC(Weighted Model Counting)의 한계점:
  - 확률과 같은 양의 실수값만을 갖는 세미링(semiring)에 제한됨.
- Algebraic Model Counting (AMC)(Kimmig, Van den Broeck, De Raedt 2017):
  - 임의의 교환반환 세미링(commutative semiring)을 사용하여 의미 있는 계수 계산 문제 정의.
  - (지식 컴파일된) 논리 공식에서 활용 가능.

- **정의 1: 교환반환 세미링 (commutative semiring)**  
  세미링 $$S = (A, \oplus, \otimes, e_\oplus, e_\otimes)$$은 다음 조건을 만족하는 대수 구조:  
  1) $$A$$는 도메인 원소들의 집합.  
  2) 덧셈 $$\oplus$$, 곱셈 $$\otimes$$는 이항 연산으로, 모두 결합법칙과 교환법칙을 만족함.  
  3) 곱셈 $$\otimes$$는 덧셈 $$\oplus$$에 대해 분배법칙을 만족함.  
  4) $$e_\oplus \in A$$는 덧셈의 항등원(중립원).  
  5) $$e_\otimes \in A$$는 곱셈의 항등원.  
  6) $$e_\oplus$$는 곱셈의 영원소(annihilator).

- **정의 2: Algebraic Model Counting (AMC)**  
  - $$S = (A, \oplus, \otimes, e_\oplus, e_\otimes)$$: 교환반환 세미링  
  - $$T$$: 명제 논리 이론  
  - $$\alpha: L \rightarrow A$$: 논리식 변수의 리터럴 집합 $$L$$을 세미링 도메인 $$A$$의 값에 매핑하는 레이블링 함수  
  - AMC 문제:  
    $$  
    \mathrm{AMC}(S, T, \alpha) = \bigoplus_{m \in M(T)} \bigotimes_{\ell \in m} \alpha(\ell),  
    $$  
    여기서 $$M(T)$$는 $$T$$의 모델들의 집합이며, $$\bigotimes_{\ell \in m} \alpha(\ell)$$는 모델 $$m$$에 대해 참인 리터럴들의 값의 곱셈 연산 결과.

- AMC가 적용되는 인공지능 문제들 예시:
  - 그래디언트 계산(gradient semiring 사용): 학습(task)에서 유용 (Manhaeve et al. 2018)
  - 불확실성 하에서의 의사결정: 최대 기대효용 계산(Maximum Expected Utilities, MEUs) 문제에 AMC 적용(Derkinderen and De Raedt 2020)

- Derkinderen와 De Raedt (2020)의 MEU를 위한 AMC 확장:  
  - 리터럴 집합 $$L$$에 대한 레이블링 함수:  
    $$  
    \{\alpha(\ell) = (p_\ell, eu_\ell, D_\ell) \mid \ell \in L \} \subset A,  
    $$  
    - $$p_\ell$$: $$\ell$$의 확률  
    - $$eu_\ell$$: 기대효용(expected utility)  
    - $$D_\ell = \{\ell\}$$, 만약 $$\ell$$이 결정 리터럴(decision literal)이면, 아니면 빈집합

  - 기대효용 세미링(expected utility semiring)을 최대화 연산(maximisation)을 포함하도록 적응:  
    - 새로운 대수 구조 $$S_{meu} = (A, \oplus, \otimes, e_\oplus, e_\otimes)$$ 정의:  
      $$  
      a_1 \oplus a_2 =   
      \begin{cases}  
      \max(a_1, a_2) & \text{if } D_1 \neq D_2 \\
      (p_1 + p_2, eu_1 + eu_2, D_1) & \text{otherwise}  
      \end{cases}  
      $$  
      $$  
      a_1 \otimes a_2 = (p_1 \cdot p_2, \, p_1 \cdot eu_2 + p_2 \cdot eu_1, \, D_1 \cup D_2)  
      $$  
      $$  
      e_\oplus = (0, 0, D), \quad e_\otimes = (1, 0, \emptyset)  
      $$  
      여기서 $$a_i = (p_i, eu_i, D_i) \in A$$이며, $$D$$는 모든 결정 변수와 그 부정의 집합임.  
      $$\max(a_1, a_2)$$는 기대효용 $$\frac{eu_i}{p_i}$$가 최대인 원소 반환(정규화 위한 기술적 조정 있음).

- 참고 사항:  
  - 세미링은 두 가지 연산만 있지만, 불확실성 의사결정 문제는 최대값(max), 합(sum), 곱(product) 세 연산을 포함하여 복잡함.  
  - $$S_{meu}$$에서는 $$\oplus$$ 연산에 입력 의존(input-dependent)이 적용되어 이 문제 해결.  
  - 따라서 $$S_{meu}$$는 일반적인 의미의 세미링이 아니며, AMC 계산 시 변수 순서 제약과 함께 사용해야 함.  
  - 여러 지식 컴파일 도구에서 이 방식 지원.

- 자세한 내용은 Derkinderen and De Raedt (2020) 참조.

---

# 2.3 Representing Markov Decision Processes

- Derkinderen and De Raedt (2020)는 의사결정 문제를 해결할 수 있었지만, 그들의 기법은 결정들 간의 시간적 또는 순차적 의존성을 고려하지 않음.
- 이러한 시나리오는 일반적으로 MDPs(Markov Decision Processes)를 이용해 모델링됨 (Puterman 2009).
- MDP는 완전히 관찰 가능하고 확률적인 환경에서의 동적 의사결정 문제를 위한 형식적 모델이며, 가산적 보상을 가짐.
- MDP는 다음 요소들로 구성됨:
  - 명시적 상태 집합 $$S$$
  - 행위 집합 $$A$$
  - 보상 함수 $$R(s, a)$$
  - 확률적 전이 함수 $$T(s, a, s') = P(s' \vert s, a)$$, 현재 상태 $$s$$와 행위 $$a$$가 주어졌을 때 다음 상태 $$s'$$로 전이될 확률을 나타냄.

- MDP를 해결한다는 것은 각 상태에 대해 최적의 행위를 찾는 것으로, 모든 상태 $$s \in S$$에 대해 다음 점화식을 푸는 것과 같음:

  $$
  U(s) = \max_{a \in A} \left( R(s, a) + \gamma \sum_{s' \in S} P(s' \vert s, a) U(s') \right)
  $$

- 이 식은 잘 알려진 Bellman 방정식이며, 값 반복(value iteration) 알고리즘으로 해결할 수 있음 (Bellman 1957).
- 값 반복은 일반적으로 모든 상태에 대해 $$U(s) = 0$$ 으로 초기화한 뒤 식 3에 따라 반복적으로 $$U$$의 값을 갱신; 이 한 번의 갱신을 Bellman 업데이트라 부름.
- 감쇠 계수 $$\gamma \in [0,1)$$ 인 경우 값 반복은 최적해로 수렴하며, 각 상태에서 기대 보상이 최대가 되는 행위를 찾음.

- 그래프로 MDP를 나타낼 때는 동적 결정 네트워크(dynamic decision networks, DDN)를 사용함 — 표준 베이지안 네트워크(BN)의 동적이고 결정 이론적 확장판.

- DDN 특징:
  - BN과 마찬가지로 원형 노드로 확률 변수 표현
  - 다이아몬드형 노드로 보상 노드 표현
  - 사각형 노드로 결정 노드 표현 (MDP의 ‘행위(actions)’를 ‘결정(decisions)’이라 부름)
- MDP는 상태 $$s, s'$$를 명시적으로 모델링하는 반면, DDN은 상태를 구성하는 확률 변수들을 모델링함 (Figure 1 참조).
- DDN을 이용한 MDP 표현의 장점은 상태를 명시적 집합 대신 확률 변수들의 집합으로 모델링함으로써 표현 크기가 (잠재적으로) 지수적으로 감소할 수 있음. 이는 평탄화된 상태 인코딩(flattened state encoding)에 비해 베이지안 네트워크의 공간 효율성과 유사함 (Russell and Norvig 2020, Section 17.1.4).
- 시간 $$t+1$$에서 상태를 베이지안 네트워크로 표현하는 것은 인수 분해된 MDP(factored MDP)를 사용하는 것과 동일함 (Boutilier, Dearden, and Goldszmidt 2000).
- 다음 섹션에서 이 구조를 적극 활용할 예정.

---

# 3 Dynamic Decision Circuits

- KC(Knowledge Compilation)의 주요 장점은 계산 비용이 많이 드는 컴파일 단계(compilation step)를 여러 번의 회로 평가(circuit evaluation)에 걸쳐 분산하여 처리할 수 있다는 점에 있다.
- 동적 결정 네트워크(DDN)의 구조가 시간에 따라 변하지 않는다고 가정할 때, 한 번만 시간 단계 간 전이를 나타내는 논리식(logic formula)을 컴파일하고 이 컴파일된 회로를 반복적으로 사용하여 DDN 문제를 해결할 수 있다.
- 이 아이디어는 Vlasselaer et al. (2016)의 연구에서 영감을 받았으며, 동적 베이지안 네트워크의 전이 함수를 전이 회로(transition circuit)로 컴파일함으로써 효율적인 필터링을 수행한다.
- 본 연구에서는 필터링 과정에 KC를 적용하는 대신, 벨만 방정식(Bellman equation, 식 3)을 KC하기 위한 새로운 인코딩 방식을 도입하였다.
- 구체적으로, DDN 내 전이 구조를 회로로 컴파일하여 Smeu 대수 구조(Section 2.2)를 이용한 평가가 벨만 업데이트에 해당하게 만든다.
- 회로의 평가는 세 가지 연산(최대(max), 합(sum), 곱(product))을 포함하는 반면, Vlasselaer et al. (2016)은 두 가지 연산(합, 곱)만 포함한다.
- 벨만 업데이트를 표현하기 위해서는 식 3의 모든 항목이 DDN 내 그래픽 요소로 대응되어야 한다.
- 식 3과 그림 1의 DDN을 비교해 보면, 항목 $$R(s, a)$$ 와 $$P(s' \vert s, a)$$ 는 그래픽적으로 대응하지만, 미래로부터 누적 기대 보상인 $$U(s')$$ 에 해당하는 그래픽 요소는 없다.
- 이를 해결하기 위해, 미래의 보상을 나타내는 유틸리티 노드 $$U$$ 를 DDN에 추가한다.
- 그림 3은 그림 1의 DDN에 이 유틸리티 노드를 추가하여 확장한 예시를 보여준다.

---

# 3.1 Encoding

- DDN(Dynamic Decision Networks)를 부울(불리언) 랜덤 변수로 가정하고 이를 명제 논리 공식으로 인코딩하는 방법을 설명.
- 다중 값(multi-valued) 랜덤 변수에도 Chavira and Darwiche (2008)의 인코딩을 사용해 확장 가능.

- 인코딩은 부울 인디케이터 변수 λ와 부울 파라미터 변수 θ로 구성.

## 인디케이터 절(indicator clauses) 생성
- 변수 Y가 비불리언이고 도메인이 {y1, ..., yn}일 때:
  - 변수 Y의 값 중 정확히 하나를 선택한다는 의미를 표현.
  - 수식:
    $$
    \left(\lambda_{y_1} \lor \ldots \lor \lambda_{y_n}\right) \wedge \bigwedge_{i<j} (\neg \lambda_{y_i} \lor \neg \lambda_{y_j})
    $$
- 변수 X가 불리언인 경우 단순히 λx로 인코딩하며, λx가 참일 때 변수 X도 참임을 의미.

## 파라미터 절(parameter clauses) 생성
각 θ 변수는 DDN의 확률, 보상, 유틸리티 테이블의 한 값에 대응.

- 확률 (Probabilities): $$ P(x'_i \vert pa_1, \ldots, pa_n) $$
  - 수식:
    $$
    \lambda_{pa_1} \wedge \ldots \wedge \lambda_{pa_n} \wedge \theta_{x'_i \vert pa_1, \ldots, pa_n} \leftrightarrow \lambda_{x'_i}
    $$

- 보상 (Rewards): $$ R(x_1, \ldots, x_n) = r_i $$
  - 수식:
    $$
    \lambda_{x_1} \wedge \ldots \wedge \lambda_{x_n} \wedge \lambda_{r_i} \leftrightarrow \theta_{r_i}
    $$

- 결정 (Decisions):
  - 수식:
    $$
    \lambda_{d_i} \leftrightarrow \theta_{d_i}
    $$

- 유틸리티 (Utilities): $$ U(x'_1, \ldots, x'_n) = u_i $$
  - 수식:
    $$
    \lambda_{x'_1} \wedge \ldots \wedge \lambda_{x'_n} \wedge \lambda_{u_i} \leftrightarrow \theta_{u_i}
    $$

## 예시: Monkey DDN 인코딩

- 의사결정 변수 (M): 
  $$
  \lambda_m \leftrightarrow \theta_m
  $$

- 보상 변수 (R):
  - 독립성 표현:
    $$
    \lambda_{r_1} \lor \lambda_{r_2} \lor \lambda_{r_3}
    $$
    $$
    \neg \lambda_{r_1} \lor \neg \lambda_{r_2}, \quad \neg \lambda_{r_1} \lor \neg \lambda_{r_3}, \quad \neg \lambda_{r_2} \lor \neg \lambda_{r_3}
    $$
  - 매핑:
    $$
    \lambda_h \wedge \lambda_{r_1} \leftrightarrow \theta_{r_1}, \quad \lambda_b \wedge \lambda_{r_2} \leftrightarrow \theta_{r_2}, \quad \lambda_m \wedge \lambda_{r_3} \leftrightarrow \theta_{r_3}
    $$

- 상태 전이 H′:
  $$
  \lambda_{h'} \leftrightarrow (\lambda_h \wedge \theta_{h' \vert h}) \lor (\neg \lambda_h \wedge \lambda_m \wedge \theta_{h' \vert \neg h, m}) \lor (\neg \lambda_h \wedge \neg \lambda_m \wedge \theta_{h' \vert \neg h, \neg m})
  $$
  - 주의: $$\theta_{h'|h,m}$$ 과 $$\theta_{h'|h,\neg m}$$ 는 문맥 특수 독립성을 이용해 $$\theta_{h'|h}$$ 로 병합됨.

- 상태 전이 B′:
  $$
  \begin{aligned}
  \lambda_{b'} \leftrightarrow &(\lambda_h \wedge \lambda_{h'} \wedge \theta_{b' \vert h, h'}) \lor (\neg \lambda_h \wedge \neg \lambda_{h'} \wedge \theta_{b' \vert \neg h, \neg h'}) \lor \\
  &(\lambda_h \wedge \neg \lambda_{h'} \wedge \theta_{b' \vert h, \neg h'}) \lor (\neg \lambda_h \wedge \lambda_{h'} \wedge \theta_{b' \vert \neg h, h'}) \lor \\
  &(\lambda_b \wedge \theta_{b' \vert b}) \lor (\neg \lambda_b \wedge \theta_{b' \vert \neg b})
  \end{aligned}
  $$

- 유틸리티 변수 (U):
  - 독립성 표현:
    $$
    \lambda_{u_1} \lor \lambda_{u_2} \lor \lambda_{u_3} \lor \lambda_{u_4}
    $$
    $$
    \neg \lambda_{u_i} \lor \neg \lambda_{u_j} \quad \text{(i < j, 모든 조합)}
    $$
  - 매핑:
    $$
    \lambda_{h'} \wedge \lambda_{b'} \wedge \lambda_{u_1} \leftrightarrow \theta_{u_1}
    $$
    $$
    \lambda_{h'} \wedge \neg \lambda_{b'} \wedge \lambda_{u_2} \leftrightarrow \theta_{u_2}
    $$
    $$
    \neg \lambda_{h'} \wedge \lambda_{b'} \wedge \lambda_{u_3} \leftrightarrow \theta_{u_3}
    $$
    $$
    \neg \lambda_{h'} \wedge \neg \lambda_{b'} \wedge \lambda_{u_4} \leftrightarrow \theta_{u_4}
    $$

---

# 3.2 Labelling

- 리터럴(literal)을 집합 $$ S_{meu} $$의 원소로 사상하는 라벨링 함수 도입 (섹션 2.2 참조)
- 각 리터럴 $$ \ell $$에 대해 3중 라벨 $$(p_{\ell}, eu_{\ell}, D_{\ell})$$ 정의
  - $$p_{\ell}$$: 확률 (probability)
  - $$eu_{\ell}$$: 기대 효용 (expected utility)
  - $$D_{\ell}$$: 결정들의 집합 (set of decisions)
- 이를 통해 DDN(Decision Diagram Networks) 문제를 AMC(Algebraic Model Counting)으로 해결 가능

## 라벨링 규칙

1. **지시 변수(indicator variables)**  
   모든 지시 변수는 중립 원소 $$ e_\otimes $$에 해당하는 라벨로 초기화  
   $$
   \alpha(\lambda_\ell) = \alpha(\neg \lambda_\ell) = (1, 0, \emptyset) \tag{9}
   $$

2. **파라미터 변수(parameter variables)**  
   - **확률(probabilities)**:  
     확률 $$ p = P(x'_i \vert pa_1, \ldots, pa_n) $$ 에 대해  
     $$
     \alpha(\theta_{x'_i \vert pa_1, \ldots, pa_n}) = (p, 0, \emptyset) \tag{10}
     $$
     $$
     \alpha(\neg \theta_{x'_i \vert pa_1, \ldots, pa_n}) = (1 - p, 0, \emptyset) \tag{11}
     $$

   - **보상(rewards)**:  
     $$ R(x_1, \ldots, x_n) = r_i $$ 인 경우  
     $$
     \alpha(\theta_{r_i}) = (1, R(x_1, \ldots, x_n), \emptyset) \tag{12}
     $$
     $$
     \alpha(\neg \theta_{r_i}) = (1, 0, \emptyset) \tag{13}
     $$

   - **결정(decisions)**:  
     $$
     \alpha(\theta_d) = (1, 0, \{d\}), \quad \alpha(\neg \theta_d) = (1, 0, \{\neg d\}) \tag{14}
     $$

   - **효용(utilities)**:  
     $$ U(x'_1, \ldots, x'_n) = u_i $$ 인 경우  
     $$
     \alpha(\theta_{u_i}) = (1, U(x'_1, \ldots, x'_n), \emptyset) \tag{15}
     $$
     $$
     \alpha(\neg \theta_{u_i}) = (1, 0, \emptyset) \tag{16}
     $$

     - 식 (15)는 벨만 방정식(Bellman equation)의 재귀적 특성을 반영함
     - 가치 반복(value iteration)에서는 효용 라벨 초기화가 아래와 같이 수행됨  
       $$
       \alpha(\theta_{u_i}) = (1, 0, \emptyset) \tag{17}
       $$

## 예시 5 (원숭이 인코딩 라벨링)

- 예시 4의 인코딩에 대한 라벨링 함수 제공
- 모든 지시 변수는 식 (9)에 따라 설정
- 결정 파라미터:  
  $$
  \alpha(\theta_m) = (1, 0, \{d\}), \quad \alpha(\neg \theta_m) = (1, 0, \{\neg d\})
  $$
- 보상 파라미터:  
  $$
  \alpha(\theta_{r_1}) = (1, -10, \emptyset), \quad \alpha(\theta_{r_2}) = (1, -4, \emptyset), \quad \alpha(\theta_{r_3}) = (1, -1, \emptyset)
  $$
  $$
  \alpha(\neg \theta_{r_1}) = \alpha(\neg \theta_{r_2}) = \alpha(\neg \theta_{r_3}) = (1, 0, \emptyset)
  $$
- $$H'$$ 관련 확률들:  
  $$
  \alpha(\theta_{h' \vert h}) = (0.2, 0, \emptyset), \quad \alpha(\neg \theta_{h' \vert h}) = (0.8, 0, \emptyset)
  $$
  $$
  \alpha(\theta_{h' \vert \neg h, m}) = (0.5, 0, \emptyset), \quad \alpha(\neg \theta_{h' \vert \neg h, m}) = (0.5, 0, \emptyset)
  $$
  $$
  \alpha(\theta_{h' \vert \neg h, \neg m}) = (0.8, 0, \emptyset), \quad \alpha(\neg \theta_{h' \vert \neg h, \neg m}) = (0.2, 0, \emptyset)
  $$
- $$B'$$에 대한 라벨링도 유사하게 생성 가능
- 효용 파라미터는 초기에 식 (17)에 의해 설정됨

---

# 3.3 Compiling

- DDN(동적 의사결정 네트워크)을 명제 논리 공식으로 인코딩하고 적절한 라벨링 함수가 주어지면, 회로를 컴파일할 수 있음.
- 이를 위해 기존의 지식 컴파일러(knowledge compiler)를 사용할 수 있으며, 컴파일된 구조의 잎 노드들을 라벨링하고 AMC(산술 모델 카운팅)을 사용하여 회로를 평가함.
- 최대 기대 효용은 다음 식으로 표현됨:
  
  $$ U(x) = AMC(S_{meu}, \Delta, \alpha \vert x) $$
  
  여기서,
  - $$\Delta$$는 컴파일된 회로를 나타내며,
  - $$\alpha \vert x$$는 상태 변수 $$X$$를 값 $$x$$로 인스턴스화한 것을 의미함.
- 회로 내에서는 이는 지시자 변수(indicator variables)의 가중치를 변경하는 것과 같음.
  - 예를 들어, 원숭이 문제(monkey example)에서 $$H = h$$ 조건화 시 $$\alpha(\neg \lambda_h) = (0, 0, \emptyset)$$로 설정함.
- **정의 3 (의사결정 회로, Bhattacharjya와 Shachter, 2007)**  
  의사결정 회로는 루트가 있는 방향성 비순환 그래프(DAG)로서 잎 노드는 변수 또는 상수로 라벨링되고, 나머지 노드는 합산(summation), 곱셈(multiplication), 최대화(maximisation) 노드임.
- **정의 4 (동적 의사결정 회로, Dynamic Decision Circuit, DDC)**  
  DDC는 재귀적 라벨링 함수를 갖는 의사결정 회로로서, 각 잎 노드는 자신이 평가되는 회로 자체의 결과에 의존하는 값을 가짐.
- 식 18을 통해 현재 시점에서의 최대 기대 효용을 계산할 수 있으나, 미래에서 오는 효용 $$U(s')$$를 알고 있어야 함.
- 다음 섹션에서 이 AMC 호출을 재귀적으로 수행하는 방법을 제시함.

---

# 4 mapl-cirup

- DDN에서 (동적 의사결정) 회로 ∆를 얻는 방법과, 레이블링 함수 α를 사용하여 대수적 모델 카운팅(AMC)으로 이를 평가하는 방법을 설명함.
- 기존의 가치 반복 알고리즘을 변형한 방법을 제시하는데, 여기서 Bellman 업데이트를 ∆에 대한 AMC 호출로 대체함.

---

# 4.1 Bellman Update Using Circuits

- Algorithm 1은 mapl-cirup의 가치 반복(value iteration) 접근 방식을 보여준다.
- 초기에는 모든 상태의 값이 0으로 설정된다 (라인 4).
- 이후 수렴할 때까지 반복적으로 Bellman 업데이트를 수행한다 (라인 5~11).
- Bellman 업데이트 (수식 3)는 AMC 호출로 대체되며, 이는 수식 18에 해당한다 (라인 7).
- 고전적인 가치 반복과 마찬가지로, 각 상태 $$s \in S$$, 즉 가능한 모든 변수 할당 $$x$$에 대해 업데이트를 수행한다 (라인 6).
- 새로 계산된 값 $$U'(x)$$는 다음 업데이트 단계에서 유틸리티 레이블 $$\alpha(\theta_u)$$를 갱신하는 데 사용된다 (라인 8).
- 그림 4는 mapl-cirup가 고전적인 가치 반복 알고리즘과 어떻게 다른지 직관적으로 보여준다.
  - 상태에 대한 반복(loop)은 명시적이지만 Bellman 업데이트는 DDC(결정 다이어그램)로 인코딩되어 있다.
- 회로의 컴파일 비용은 총 $$|VI|^2 |X|$$ 단계에 걸쳐 분산된다.
    - 여기서 $$|VI|$$는 최적 정책에 수렴하는 데 필요한 반복 횟수,
    - $$|X|$$는 입력으로 주어진 동적 결정 네트워크(DDN)의 상태 변수 개수이다.

### Algorithm 1: Value Iteration with DDCs

```
1:  inputs: the DDC ∆, the labelling function α, the convergence error to terminate ϵ
2:  local variables: U, U′, vectors of utilities for states x; δ infinite norm of change in utilities
3:  procedure mapl-cirup(∆, α, ϵ):
4:      U ← 0
5:      repeat
6:          for each instantiation x do
7:              U′(x) ← AMC(Smeu, ∆, α|x)
8:              α(θu) ← U′
9:          δ ← ||U′ − U||
10:         U ← U′
11:     until δ < ϵ
12:     return U
```

- Theorem 1 (mapl-cirup의 정확성):
  - Algorithm 1은 DDC ∆과 레이블 함수 α로 인코딩된 문제의 최적 해를 올바르게 계산한다.
- 증명 개요:
  - 가치 반복의 수렴성이 이미 증명되어 있으므로,
  - AMC 프레임워크를 Bellman 업데이트 수행에 정확히 사용하는지만 증명하면 된다.
  - Smeu (Derkinderen & De Raedt 2020) 정의를 사용하고,
  - 레이블 함수는 Smeu에서 설명한 함수의 직접 적용이다.
  - DDN 파라미터를 적절한 레이블에 매핑하고,
  - 유틸리티 파라미터 $$U$$를 Bellman 업데이트 (수식 3)에 따라 수정한다.
  - 상태와 유틸리티 파라미터 간의 상응 관계는 인코딩에 의해 보장된다.
  - 나머지 인코딩은 Chavira와 Darwiche(2008)의 인코딩을 적절히 변형한 것이다.

- mapl-cirup의 계산적 단점:
  - 각 상태마다 하나의 유틸리티 파라미터 $$U$$를 도입함으로써,
  - 컴파일된 표현 ∆의 변수 수와 크기가 증가한다.
  - 또한, 모든 명시적 상태에 대해 반복해야 한다.
  - Hoey et al. (1999)와 유사한 방식으로 유틸리티 함수를 더 컴팩트하게 표현하는 연구가 미래 과제로 제시된다.
  
- 주요 장점:
  - 회로 ∆는 단 한 번만 컴파일되며,
  - 반복되는 시간 구조를 효과적으로 활용하여 컴파일 비용을 여러 반복에 분산시킨다.
  - ∆는 가치 반복 과정 전체에서 변하지 않으므로,
  - 가치 반복 시작 전에 더 컴팩트한 표현을 얻기 위해 계산 자원을 더 투자하는 것이 유리하다.

---

# 4.2 Learning

- mapl-cirup는 compile+evaluate 패러다임을 따르기 때문에, 대수적 모델 카운팅(algebraic model counting) 프레임워크 내에서 개발된 다양한 기법에 접근할 수 있음.
- 따라서, 서킷 ∆를 재사용하여 보상 파라미터(reward parameters)를 학습하는 작업에 적용 가능.
- 주어진 데이터 셋 E는 궤적 τ = ⟨s0, a0:k, r0:k⟩들의 집합이며, 각 궤적은 다음으로 구성됨:
  - 초기 상태 s0 (알려진 상태)
  - 이 상태에서 취한 연속적인 k+1개의 행동 a0:k
  - 각 행동 이후 획득한 보상 r0:k
- 대응하는 DDN에는 각 변수가 미지의 보상 파라미터를 가질 수 있으며, 보상은 상태 변수에 연관된 (미지의) 보상의 합성 함수임.
- 학습 과제는 중간 상태들이 관측되지 않은 상황에서 이 미지의 보상 파라미터를 학습하는 것임.
- 이를 위해, ∆ 위에서 그래디언트 기반 접근법을 사용하며 대수적 프레임워크에 기반한 그래디언트 계산을 활용.
- 평균 제곱 오차 손실 함수(mean squared error loss)를 도입:
  
  $$
  \frac{1}{|E|}\sum_{\tau \in E} \sum_{t=0}^k \left( ce_{ut;\theta}(s_0, a_{0:t}) - r_t \right)^2 \tag{19}
  $$

- 여기서,
  
  $$
  ce_{ut;\theta}(s_0, a_{0:t}) = \sum_{s_t} P(s_t \vert s_0, a_{0:t}) R_\theta(s_t, a_t) \tag{20}
  $$

- $$R_\theta$$는 학습 가능한 파라미터 θ로 파라미터화된 보상 함수임.
- $$ce_{ut;\theta}(s_0, a_{0:t})$$는 시간 t에서의 기대 효용(expected utility)으로, 초기 상태와 t까지의 행동이 주어졌을 때 현재 파라미터 θ에 따른 값임.
- 이 손실 함수를 사용함으로써 기대 효용과 실제 관측된 보상 $$r_t$$ 간의 차이를 최소화함.
- 이 손실 함수는 이미 컴파일된 ∆로부터 쉽게 계산 가능함.
- 추가로, Gutmann et al. (2008)의 방법을 통합하여 확률 파라미터도 동시에 학습할 수 있음.
- 이를 위해 보상과 결정(decision)을 고려하며, DDC 위에 기대 효용 세미링(expected utility semiring)을 사용함.

---

# 5 Related Work

- SPUDD (Hoey et al. 1999)은 고전적인 가치 반복(value iteration) 알고리즘의 변형으로, 지식 컴파일(knowledge compilation)을 활용해 벨만 업데이트(Bellman update)를 대수적 결정 다이어그램(ADDs, Algebraic Decision Diagrams)으로 수행한다.
  - ADDs는 공통되는 값을 활용하여 컴팩트한 표현을 제공하며, 곱셈과 덧셈 연산을 지원한다.
  - SPUDD는 가치 반복 과정에서 여러 번 컴파일 작업을 수행한다.
  - 반면, mapl-cirup은 한 번만 컴파일 후 같은 다이어그램 $$\Delta$$를 여러 번 재사용한다.
  - 또한 mapl-cirup은 SPUDD가 다루지 않는 파라미터 학습(parameter learning)을 동일한 $$\Delta$$ 위에서 수행한다.

- 비록 SPUDD가 20년 이상 전에 소개되었으나, 여러 최신 근사 기법 연구(예: Hayes et al. 2021; Heß et al. 2021; Moreira et al. 2021; Dudek et al. 2022; Tan and Nejat 2022)에서 여전히 완전한 사실화 MDP(factored MDP)를 정확히 푸는 최첨단(state-of-the-art) 방법으로 간주되고 있다.

- Vlasselaer et al. (2016)은 동적 베이지안 네트워크(Dynamic Bayesian Networks)가 지식 컴파일 기술로부터 어떻게 이득을 얻는지 조사했지만,
  - 의사 결정 이론(decision-theoretic) 설정이나 학습 작업은 다루지 않았다.

- Derkinderen와 De Raedt (2020)은 비회귀 문제(non-temporal)에 한정하여 회로(circuit)를 사용한 대수 모델 카운팅(algebraic model counting)으로 기대 효용(expected utilities)을 계산하고 의사 결정을 최적화했다.
  - 본 연구는 시간에 걸쳐 의사 결정을 고려하는 설정임.

- 최근에 도입된 반복 합-곱-최대 회로(Recurrent Sum-Product-Max networks, Tatavarti et al. 2021)는
  - 모형으로부터 컴파일되는 것이 아니라 데이터로부터 직접 구조와 파라미터를 학습한다.
  - 완전히 관찰된 경로로부터 누적 보상을 신호로 사용하여 학습하므로, 본 연구의 학습과는 다른 과제이다.
  - 또한 해당 접근법은 정확한 벨만 업데이트를 수행하지 않는 것으로 이해된다.
  - 변수별로 유틸리티 값을 업데이트하며 모든 명시적 상태를 고려하지 않아, 선형 근사(linear approximation)(Guestrin et al. 2003)과 더 유사하다.

- 표 1은 mapl-cirup과 SPUDD 간의 비교를 보여주며,
  - $$|\Delta|$$: 회로 크기(노드 총수)
  - KC: 컴파일 시간
  - VI: 최적 솔루션 탐색 시간
  - SPUDD는 컴파일 시간을 제공하지 않고, VI 시간을 0.01초 단위로 보고한다.

---

# 6 Experiments

- **실험 환경**  
  - Intel CPU E3-1225v3 @ 3.20 GHz, 32GB 메모리 사용  
  - 모든 실험은 10번 수행 후 평균 실행 시간 보고 (분산은 무시)  
  - mapl-cirup의 하이퍼파라미터: 할인 인자 $$\gamma = 0.9$$, 허용 오차 $$\epsilon = 0.1$$  
  - 총 실행 시간 제한(timeout) 600초 적용 (그림에서 점선으로 표시)  
  - 구현에 PySDD 패키지 사용, Python3 성능 문제 완화를 위해 Numba를 사용하여 Bellman 업데이트 회로 JIT 컴파일  
  - 비교 대상: C++로 구현된 SPUDD (버전 3.6.2), 동일 하이퍼파라미터 적용  

- **Q1: mapl-cirup의 일반적인 성능 평가**  
  - elevator, coffee, factory (SPUDD 저장소 MDP 인스턴스)와 monkey 인스턴스,  
    그리고 변수 수 $$|X|$$에 따라 파라메트릭한 두 DDN(families) 설계:  
    - cross-stitch 구조  
    - chain 구조 (의존성 연쇄로 인해 지수적 복잡성 유발)  
  - 목표: 내부 상태 변수 간 의존성 구조가 mapl-cirup 및 SPUDD에 미치는 영향 조사  
  - baseline 및 정확도 검증 위해 SPUDD와 SPUDDisd (SPUDD의 향상 버전, 복잡한 인스턴스에 적합)와 비교  
  - 결과:  
    - mapl-cirup와 SPUDD 모두 본질적 어려움으로 인한 지수적 폭발 문제 경험  
    - mapl-cirup이 한 시점만 knowledge compile하여 전체 $$\Delta$$ 회로 크기가 훨씬 작음  
    - SPUDD는 각 반복마다 새 회로 컴파일로 총 노드 수가 더 많음  
  - 관련 표 및 그림: Table 1, Figure 5  

- **Q2: 유틸리티 함수의 지수적 표현이 성능에 미치는 영향**  
  - mapl-cirup의 단점: 유틸리티 함수를 회로 내에 명시적으로 포함 (Section 4.1 참조)  
  - 개선 시도로 잘 알려진 선형 근사법 적용 (Guestrin et al., 2003)  
    - 유틸리티 노드 수를 $$2^{|X|}$$ 에서 $$|X|$$ 로 축소  
  - 실험 결과:  
    - 근사 유틸리티 함수 $$\Delta^{\approx}$$가 회로 크기 및 실행 시간에 긍정적 영향 미침  
    - 구조를 활용하는 컴파일 접근법과 미분 가능한 학습과 호환 가능함을 시사  
  - 관련 표 및 그림: Table 2, Figure 7  

- **Q3: 손실 함수가 보상 함수 학습에 대한 좋은 지표인가?**  
  - 평가 지표:  
    1. 보상 파라미터에 대한 평균 상대 오차  
    2. 각 상태 보상에 대한 평균 상대 오차 (계획에 중요)  
  - 학습 방법:  
    - 기대 유틸리티 세미링 사용해 손실 함수 계산  
    - TensorFlow 자동 미분 이용하여 기울기 추출  
    - Adam optimizer 사용 (학습률 $$\alpha = 0.1$$, $$\hat{\epsilon} = 10^{-7}$$ 등 기본값)  
    - 보상 파라미터 초기값: 정수 구간 $$[-30, 30]$$에서 균등 샘플링  
    - 데이터셋: 길이 5인 100개의 궤적 (batch size 10)  
  - 실험 예제: coffee 사례에 추가 보상 파라미터 추가하여 난이도 상승  
    - 보상 파라미터는 정수 구간 $$[1, 10]$$에서 샘플링  
    - 보상 분포는 미지로 가정  
  - 결과:  
    - 10회 반복 실험에서 평균 및 표준 편차 보고 (Figure 8)  
    - 손실 감소와 함께 파라미터 및 상태 오차 감소 확인  
    - 상대 상태 오차가 평균 2.94에서 0.41로 유의미하게 감소  
    - 상대 파라미터 오차는 다소 높으나, 보상 함수의 가법적 특성과 표현 자유도 때문으로 해석  
  - 추가 실험 결과는 부록 Section B 참고

---

# 7 Conclusions and Future Work

- 본 논문에서는 동적 의사결정 회로(Dynamic Decision Circuits, DDC)와 마르코프 계획(Markov Planning)을 DDC 내 추론 문제로 환원하는 가치 반복 기반 알고리즘인 mapl-cirup을 소개하였다.
- compile+evaluate 패러다임 덕분에, 컴파일된 다이어그램을 대수적 모델 카운팅(algebraic model counting) 프레임워크 내에서 다양한 태스크 해결에 활용할 수 있음을 보여주었다.
- 특히, 오프라인 강화학습 방식으로 궤적(trajectories)에서 보상 파라미터를 학습하는 학습 과제를 정의하고 해결하였다.
- 본 접근법은 인자분해 표현(factored representations)을 넘어서며, 정확한 추론 방법과 정책 경사 강화학습(policy gradient reinforcement learning)과 같은 근사 접근법을 통합하는 출발점이 된다.
- 향후 연구 방향으로는 다음을 고려한다:
  - SPUDD에서 사용되는 ADD 연산을 통합하여 유틸리티 함수 표현을 보다 효율적으로 만드는 방법 연구; 이는 계산 비용을 완화하면서도 방법의 정확성을 유지한다.
  - St-Aubin, Hoey 및 Boutilier (2000)의 근사 방법을 mapl-cirup과 결합하여 실행 시간 개선 및 더 큰 도메인으로의 확장 가능성 탐색.
  - 컴파일되고 미분 가능한 표현을 민감도 분석(sensitivity analysis) 등 다른 태스크에 활용하는 방안 연구.

- 부록 A에서는 intra-state dependency(시간 동일 슬라이스 내 상태 변수 간 의존성) 처리 문제와 관련하여:
  - 일부 기존 방법(SPUDD)은 이 의존성을 다루기 위해 상태 변수 분할 등의 비효율적 변환이 필요하지만, mapl-cirup은 지식 컴파일을 활용해 이러한 복잡한 의존성을 직접 처리할 수 있어 지수적 폭발 문제를 완화한다.
  - SPUDD와 SPUDD-isd(확장버전)를 구분하며, SPUDD-isd는 intra-state dependency를 처리할 수 있다.

- 부록 B에서는 reward parameter 학습 성능을 추가 실험으로 평가:
  - 전이 함수가 더 확률적일 때 (Figure 9a), 학습된 파라미터 품질은 약간 저하되나 분산이 증가함.
  - 데이터셋 크기를 100개 궤적에서 10개로 줄여 학습했을 때 (Figure 9b)도, 더 많은 학습 Epoch가 필요하고 분산도 커지지만 파라미터 품질은 크게 손상되지 않음을 확인함.

- 핵심 수식 예시: 선형 가치 함수는 기저 함수 집합 $$H=\{h_1,\ldots,h_k\}$$과 가중치 벡터 $$w=(w_1,\ldots,w_k)'$$에 대해  
  $$V(x) = \sum_{j=1}^k w_j h_j(x)$$  
 로 표현된다.
