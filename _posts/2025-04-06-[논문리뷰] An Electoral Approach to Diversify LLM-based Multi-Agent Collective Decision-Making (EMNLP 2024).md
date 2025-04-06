---
title: "[논문리뷰] An Electoral Approach to Diversify LLM-based Multi-Agent Collective Decision-Making (EMNLP 2024)"
date: 2025-04-06 00:00:00 +0900
categories:
  - Paper Review
tags:
  - EMNLP 2024
  - Multi-agent
---

현대의 대형 언어 모델들은 복잡한 문제 해결 및 집단 의사결정에서 협력적 시너지를 보여주고 있으며, 본 연구에서는 52개의 시스템을 조사하여 의사결정 방식의 다양성이 부족함을 발견했습니다. 이를 개선하기 위해 다양한 순위 선호 투표 메커니즘을 포함한 GEDI라는 모듈을 제안하며, 이 방법이 LLM의 추론 능력과 강건성을 향상시킨다는 것을 실증 연구를 통해 입증하였습니다.

---

# 1 Introduction

- 다중 에이전트 시스템은 대규모 언어 모델(LLMs)의 등장 이전부터 주목받음.
- 최근 LLM의 발전으로 LLM 기반 에이전트에 대한 관심 급증.
- 효과적인 프롬프트 엔지니어링과 에이전트 상호 작용 방식 등이 협력 LLM 에이전트 연구 촉진.
- 다양한 환경에서 LLM 기반 에이전트가 배치됨:
  - 소규모 커뮤니티 시뮬레이션
  - 법원 판결 예측
  - 디지털 아바타 생성
  - 대화 기반 게임 참여 등.
- 기존 연구는 에이전트 간 의사소통 및 상호작용에 집중, 집합적 의사결정(CDM) 측면은 소홀함.
- 52개의 최근 LLM 협력 시스템 분석 결과:
  - 결정은 ‘독재자’ 에이전트에 의해 이루어지거나 단순 다수결 투표에 의존.
  - 한 사례는 공리주의 접근 방식을 채택함.
- CDM 방법을 사회 선택 이론 관점에서 검토하며 근본적인 기준 미달성을 지적:
  - 독재적 방법은 단일 에이전트에 의존하여 취약.
  - 다수결 투표는 독립성 및 콩도르세 기준을 충족하지 않음.
  - 공리주의는 다수 및 콩도르세 기준 위반.
- 이러한 기준 위반은 LLM 기반 에이전트 간 개인 선호에서 집합 결정으로의 전환을 방해할 수 있음.
- Arrow의 정리에 따르면 완벽한 투표 기반 CDM 시스템 설계는 불가능하지만, 다양한 CDM 방법을 통합함으로써 일부 한계 극복 가능.
- 이를 위해 새로운 선거 CDM 모듈 GEDI 개발.
- CDM 방법의 잠재적 영향을 평가하기 위해 세 가지 다중 선택 질문-응답(MCQA) 기준에서 실증 사례 연구 진행.
- 주요 발견:
  1. CDM 방법 적용 시 단일 에이전트 의사결정보다 일반적으로 더 나은 결과 도출, 다만 계산 비용 증가.
  2. 시너지의 정도는 백본 모델과 기준에 크게 의존.
  3. 대부분의 투표 방법은 효과적인 최소 정족수 필요.
  4. CDM 방법은 신뢰할 수 없는 에이전트에 대한 견고성에서 차이를 보임.
- 이러한 관측은 LLM 기반 다중 에이전트 시스템의 효과성을 평가하는 데 기여를 기대함.

---

# 2.1 Background

- 다중 에이전트 시스템은 자율적인 행동 및 상호작용 능력을 가진 여러 계산 요소(‘에이전트’)로 구성됨 (Wooldridge, 2009).
- LLM(대규모 언어 모델)의 출현 이전에도 다중 에이전트 시스템에 대한 연구는 다양한 분야에서 중심 주제였음 (Silver et al., 2017; Dorri et al., 2018).
- LLM의 빠른 발전은 LLM을 에이전트로 활용하려는 관심을 불러일으킴 (Xi et al., 2023).
- 효과적인 프롬프트 기법의 출현이 개별 LLM 에이전트의 성능을 크게 향상시킴:
  - Chain-of-Thought (Wei et al., 2023)
  - Self-Consistency (Wang et al., 2023c)
  - ReAct (Yao et al., 2023)
  - Reflexion (Shinn et al., 2023)
  - DiVeRSe (Li et al., 2023e)
- 단일 에이전트 프레임워크는 특정 NLP 작업에서 remarkable 성공을 보여주지만, 공통 감각 추론 및 장기 계획과 같은 더 복잡한 문제에서는 어려움을 겪음 (Wang et al., 2023b).
- 이에 따라, 일부 연구자들은 다중 LLM 에이전트 협력을 유망한 방향으로 제안함.

---

# 2.2 Collective Decision-Making in LLM-based Multi-Agent Collaboration

- 집단 의사결정(CDM)은 자율적 개체 집단이 결정을 내리는 과정이다.
- CDM은 동물 사회와 인간 공동체에서 일반적이며, 개인의 결정보다 더 우수한 결정이 나오는 경향이 있다.
- LLM의 발전으로 LLM 기반 다중 에이전트 시스템에서 자율적인 CDM 프로세스가 가능해졌다.
- 52개의 새로운 프레임워크 조사 결과, CDM 메커니즘이 충분히 주목받지 못하고 있음이 나타났다.
  - 대다수 시스템은 독재적인 판단에 의존하거나 다수 투표 방식으로 결정을 내린다.

## 현재 LLM 기반 다중 에이전트 시스템의 CDM 접근 방식 분류
1. **독재적 접근**
   - 단일 에이전트가 결정을 승인하는 시스템.
   - '독재자'는 다른 에이전트와 소통하고 조언을 받을 수 있음.
   - 다양한 별명이 존재하며, 이러한 에이전트는 역할을 전담한다.
   - 사례: 가상 소프트웨어 개발에서 다양한 역할의 LLM 에이전트 활용.

2. **다수 투표**
   - 가장 많은 1순위 표를 선택하는 방식.
   - 다수결 투표(절대 다수) 및 합의 도출도 포함된다.
   - 다수 토론 과정을 통해 결론 도출.
   - LLM의 사실성 및 추론 능력 개선에 기여함.
   - 특정 시나리오에 맞춰 다수 투표 방식 선택.

3. **공리적 접근**
   - 가능한 결정의 영향을 정량화하여 집단의 '효용'을 극대화하는 옵션 선택.
   - 효용은 외부에서 미리 결정되거나 업데이트된다.
   - 최근 LLM 기반 프레임워크에서는 드물지만, 이전 시스템들에서는 중요한 방법.

4. **CDM 없음 또는 명시되지 않음**
   - 일부 시나리오는 CDM이 필요하지 않음.
   - 일대일 합의가 가끔 발생할 수 있음.
   - Strict linear collaboration 혹은 분산 팀 구성 등으로 CDM 과정이 부재함.

- 다양한 CDM 방식의 부족을 인식하고, 사회적 선택 이론에서 영감을 받아 널리 사용되는 방법들의 장단점을 검토할 필요가 있다.

---

# 3 A Social Choice Theory Perspective on Collective Decision-Making

- 사회적 선택 이론은 개인의 선호에서 집단적 결정으로의 전환을 다룸.
- 인간은 고대부터 집단적 의사결정을 수행하고 발전시켜 옴.
- 현대 사회적 선택 이론은 Kenneth J. Arrow의 저서 **Social Choice and Individual Values** (1951) 출판으로 확립됨.
- Arrow의 이론은 공리적으로 포멀라이즈되며 다양한 선거 시스템을 비교 분석함.

---

# 3.1 Related Work Incorporating Social Choice Theory into NLP Research

- 관련 연구는 사회 선택 이론을 다음 분야에 통합하는 데 주로 집중됨:
  - 모델 정렬 (Mishra, 2023)
  - 모델 앙상블 (Jiang et al., 2023b)
  - 텍스트 생성 및 선호 추정 (Fish et al., 2023)

- Jarrett et al. (2023)은 공리주의적 접근을 통해 LLM 에이전트를 인간의 디지털 대표로 활용함.

- Irurozki et al. (2022) 및 Rofin et al. (2023)은 NLP 벤치마킹에서 멀티태스크 점수의 정통 평균 집계 방식의 한계를 지적하고, 사회 선택 이론에 기반한 새로운 집계 방법을 제안함:

- Wang et al. (2023c) 및 Xue et al. (2023)은 다수결 투표를 통해 여러 생성된 추론 경로에서 답변을 선택하는 방법을 제안하며, 공리주의적 접근보다 개선된 결과를 제공함.

- 최근 Li et al. (2024)은 gpt-3.5 (Ouyang et al., 2022)와 Llama-2 (Touvron et al., 2023)에서 다수결 투표의 시너지를 입증하였으나, 다른 CDM 방법과의 비교가 부족함.

- 또 다른 동시에 진행된 연구 (Yang et al., 2024)는 투표 행동 관점에서 인간과 LLM 간의 차이를 조사함.

- 그러나 이전 연구들은 LLM 기반의 멀티 에이전트 CDM 방법을 다양화하려는 우리의 주요 목표와는 겹치지 않음.

---

# 3.2 Criticism on Prevalent CDM Methods in LLM-based Multi-Agent Collaboration

- **독재적 방법**
  - 단일 에이전트가 그룹의 결정을 내림.
  - 효율적이지만, 유일한 에이전트에 대한 의존성으로 인해 편향적이고 강인함이 결여됨.

- **공리주의 및 기수 투표 방법**
  - 그룹 구성원의 개별 선호를 집계하고 공개함.
  - 외부에서 부과된 효용의 불안정성과 임의성 문제가 존재.
  - 정확한 기수 효용을 가정해야 하며, 불균형한 효용 분포가 다수결 기준을 위반할 수 있음.

- **다수결 투표**
  - 순위를 매긴 투표로 분산된 의사결정 방법의 전형적인 예.
  - 현재 LLM-에이전트 협업 프레임워크에서 다수결 투표가 선택됨.
  - 직관적으로 안전해 보이지만, 애로우의 정리에 따르면 자명한 기준을 위반함.

- **애로우의 정리**
  - IIA(무관한 대안의 독립성)와 콩도르세 기준 위반.
  - 다양한 투표 시스템은 근본적인 결함이 있음.
  
- **투표 시스템의 완벽함**
  - 완벽한 투표 시스템의 구성은 불가능함.
  - 단일 실패 지점에 빠질 위험을 줄이기 위한 필요성 존재.

- **현대 분산 투표 시스템의 도입**
  - LLM-에이전트의 자연어 기반 '판단'을 활용해야 함.
  - 기호 투표를 특히 강조하여 보다 다양한 접근을 시도해야 함.

---

# 4 Diversifying LLM-based Multi-Agent CDM

- CDM 접근 방식의 다양성을 높이기 위해 LLM-에이전트 프레임워크 내에서 다양한 CDM 방법을 통합할 것을 제안
- 인간의 사회-정치적 관행에 기반한 여러 CDM 방법을 포함
- **General Electoral Decision-making Interface (GEDI)**라는 선거 CDM 모듈을 개발
  - 여러 일반적인 서열 선호 투표 시스템 통합
- 그림 2는 GEDI와 LLM 기반 다중 에이전트 시스템에서 일반적으로 사용되는 다른 CDM 방법 간의 주요 차이점을 강조

---

# 4.1 Definition

- 다수의 대안 의사결정 과정을 고려
- 에이전트 집합: $$N = \{1, 2, ..., n\}$$
- 대안 집합: $$A = \{a_1, a_2, ..., a_m\}$$, 단 $$m \geq 2$$
- 선호 순위 투표는 대안 $$A$$의 엄격한 부분 순서 $$\succ$$로 정의
  - **전이성**: 모든 $$a, b, c \in A$$에 대해, 만약 $$a \succ b$$이고 $$b \succ c$$라면 $$a \succ c$$
  - **완전성**: 모든 $$a, b \in A$$에 대해 $$a \succ b$$ 또는 $$a \prec b$$
- 약한 순서 변형도 존재 (대안 두 개에 대한 무관심 표명 가능)
- GEDI의 입력:
  1. 프로필 $$P = (\succ_1, \succ_2, ..., \succ_n)$$: 각 유권자의 투표 모음
  2. 투표 시스템(사회적 선택 함수, SCF): $$f : L(A)^n \rightarrow C(A)$$, 엄격한 선호 집합에 대한 대안 집합을 반환
- 출력: $$f(P)$$는 대안 집합 $$A$$의 비어 있지 않은 정렬된 부분집합

---

# 4.2 Assessed Electoral Methods

- **선택된 CDM 방법**
  - 10가지 방법: Blind Dictatorial, Informed Dictatorial, Mis-informed Dictatorial, Range Voting, Plurality, Borda Count, Bucklin, Minimax, Ranked Pairs 및 랜덤 기준.

- **Dictatorial**
  - **Blind Dictatorial**
    - 한 에이전트를 임의로 선택하여 그 에이전트의 선호 순위를 결정으로 인정.
  - **Informed Dictatorial**
    - '독재자' 에이전트가 투표 결과를 검토한 후 결정.
  - **Mis-informed Dictatorial**
    - 실제 투표 대신 무작위 투표에 기초하여 독재자가 상담 받음.

- **Range Voting**
  - 에이전트들이 지정된 구간 내에서 대안에 점수를 매김.
  - 최고 점수를 받은 대안이 승리.

- **Plurality**
  - 첫 번째 선호만 고려, 이후 선호는 무시.
  - 가장 많은 1순위 투표를 받은 후보자가 승리.

- **Bucklin Voting**
  - 첫 번째 선호 투표를 먼저 집계하고, 절대 다수가 없으면 다음 선호 투표를 고려.
  - 절대 다수 후보가 나올 때까지 반복.

- **Borda Count**
  - 각 투표에서 대안의 순위에 따라 점수를 부여.
  - 표준 Borda 카운트에서는 m개의 대안 중 i번째 순위에 m - i 점수 부여.

- **Instant-Runoff Voting (IRV)**
  - 다단계 메커니즘으로, 가장 적은 1순위 투표를 받은 대안을 반복적으로 제거.
  - 제거된 대안의 투표가 살아남은 대안으로 이전됨.

- **Minimax**
  - '최소한의 최악의 비호감'을 가진 대안을 선택.
  - 함수 $f(a, b)$는 대안 $a$에 대한 $b$의 전체 '호감'을 나타냄.
  - 최악의 비호감은 $max \, f(b, a)$로 정의됨.

- **Ranked Pairs**
  - 모든 투표를 쌍별 비교로 분해하고, 빈도에 따라 정렬.
  - 가장 빈번한 쌍부터 시작하여 비교 행렬을 채움.
  - 다른 대안에 대해 모든 긍정적인 결과를 가진 대안이 승리.

---

# 5.1 Experiment Setup

- **데이터셋**
  - 본 연구의 주요 초점은 의사결정 과정에 있으며, MCQA 벤치마크가 적합.
  - 선택지는 사전에 정의되어 있음.
  - 성능 평가를 위해 MMLU, MMLUPro, ARC-Challenge 사용.
  - 참고 문헌: Park et al. (2022), Liu et al. (2023b), Zhang et al. (2023b), Google (2023), Jiang et al. (2023a).

- **백본 모델**
  - 다양한 아키텍처와 파라미터 크기의 언어 모델 기반 에이전트를 시뮬레이션하기 위해 6개의 오픈소스 모델 수집:
    - mistral-7b
    - glm-4-9b
    - llama-3-8b/70b
    - qwen-1.5-72b/110b
  - 높은 성능 모델인 gpt-3.5 및 gpt-4도 테스트에 포함.
  - 모든 모델의 온도는 0.7로 고정, OpenAI 모델은 1.0.

- **측정 및 평가**
  - 수정되지 않은 언어 모델을 테스트 에이전트로 활용.
  - 각 질문 앞에 ‘You are the {랜덤 숫자}-th rater’라는 짧은 지시 추가.
  - 의사결정 집단은 동일한 백본 모델로 구축된 에이전트로 구성됨.
  - 각 에이전트는 독립적으로 선택지의 선호 순위를 제공.
  - 모든 순위(투표)를 수집하여 프로파일을 형성, GEDI는 선택된 투표 규칙에 따라 집합적인 선호 순위를 출력.
  - ‘독재자’ 에이전트는 다른 에이전트의 투표를 받고 질문받음.
  - 프로파일 P에 10개의 선호 순위가 있으면, GEDI의 투표 시스템 $f(P)$는 모든 선택지의 정렬된 목록을 출력.
  - 첫 번째 요소가 정답과 일치하면 질문이 올바르게 답변된 것으로 간주.
  - MMLU의 원래 설정을 따라 5-shot 예시 프롬프트 사용.
  - 모든 방법은 선호 순위 형식을 취하지만, 범위 투표는 순위 외에 숫자 선호 점수 필요.

---

# 5.2 Main Results

- **5회 평균 정확도**: 결과는 테이블 2에 보고됨.
- **무작위 기준선 및 범위 투표**:
  - 무작위 기준선의 정확도는 MMLU와 ARC-Challenge에서 약 $25.0$ 정도, MMLU-Pro에서 약 $10.0$.
  - 대부분의 모델은 점수 기반 범위 투표에서 성능 저하를 보임. 예외로 llama-3-70b, gpt-3.5, gpt-4가 있음.
  
- **기본 모델 성능**:
  - 다양한 모델의 성능 비교 포함.
  - 예: llama-3-70b는 $25.3$의 정확도로, 여러 방법에서 상대적 개선을 보임.

- **독재적 방법**:
  - 독재적 방법의 색상 숫자는 블라인드 독재 대비 성능을 나타냄.
  - 정보가 제공된 경우, 대부분의 모델이 더 높은 성능을 보임. 하지만 다른 서열 방법에 비해 우수하지 않음.
  - 잘못된 정보로 인한 악화를 보여주는 증거가 있음, glm-4-9b와 gpt-4는 상대적으로 부정적인 영향을 덜 받음.

- **서열 방법**:
  - 투표 기반 서열 방법은 일반적으로 블라인드 독재보다 더 나은 정확도를 기록함.
  - 작은 모델에서 더 큰 성능 향상을 보임.
  - MMLU 벤치마크에서 glm-4-9b, gpt-3.5, gpt-4가 각각 평균 $2.9\%$, $4.9\%$, $6.5\%$ 향상됨.
  
- **전반적인 결론**:
  - 여러 CDM 방법이 모든 모델에서 긍정적인 영향을 미침. 
  - 특정 방법이 모델에 따라 약한 성능 차이를 보임, 추가적인 분석이 필요함.

---

# 5.3 Analysis and Discussion

- **최소 유효 투표 정족수**
  - 질문: 효과적인 의사결정 그룹을 구성하기 위한 최소 에이전트 수는?
  - 여러 투표 방법을 통해 에이전트 수를 늘릴 경우 정확도에서 눈에 띄는 차이를 보임.

- **투표 에이전트 수에 따른 정확도**
  - Figure 3에서 다양한 에이전트 수에 따른 정확도를 비교:
    - 두 개 이상의 에이전트 경우에서 유의미한 정확도 향상.
    - GPT 모델은 두 개의 에이전트 이상에서 정확도가 떨어짐.
    - Borda 방식은 평면에 도달하기 위해 더 많은 에이전트를 요구.
    - Range 방식은 GPT-4에서 큰 향상을 보임.

- **신뢰할 수 없는 에이전트에 대한 견고성**
  - LLM 에이전트가 잘못된 판단을 할 때의 영향 평가.
  - 그림 4는 불완전한 투표 집합의 성능을 보여줌:
    - 4명의 신뢰할 수 없는 에이전트까지 유지.
    - 공정보다 복잡한 방법이 더 불안정함.

- **Hit-Rate@K 차이**
  - Hit-rate@k는 화답의 첫 k개 선호의 누적 정확도를 나타냄.
  - 그림 5에서 다양한 투표 방법의 성능 차이를 확인:
    - Plurality가 신뢰할 수 없는 에이전트에 강하지만 최악의 선택 제거 우선 순위에서는 부족.
    - Borda와 ranked pairs는 잘못된 선택을 제외하는 데 강력함.

- **주제별 성능 개선**
  - 그림 6에서 성능 개선이 균일하지 않음을 보여줌:
    - Plurality의 정확도 향상이 -5.8%에서 +15.0%까지 다양함.
  - 그림 7에서 주제 및 CDM 방법 간 성과 차이:
    - 특정 주제에서 Plurality와 Borda Count 간의 차이 없음.

- **결론**
  - 다양한 의사결정 방법의 활용이 LLM 기반 다중 에이전트 협업에서 중요함을 확인.

---

# 6 Conclusion and Future Work

- LLM 기반 에이전트에 대한 연구가 확대되는 가운데, 52개의 다중 에이전트 협력 프레임워크를 조사함.
- CDM(집단 결정 메커니즘)의 다양성이 부족함을 발견.
- 인기 있는 CDM 방법의 한계를 사회 선택 이론 관점을 통해 분석.
- 현재 CDM 장의 다양화를 목표로 하여 인간 사회 관행에서 영감을 얻음.
- 다양한 CDM 방법을 여러 벤치마크를 통한 경험적 사례 연구에서 탐구함.
- 실험을 통해 얻은 풍부한 관찰 결과는 LLM의 집단 행동 연구에 대한 통찰을 제공.
- 본 연구는 향후 연구의 여러 길을 열어줌.
  - 특정 작업과 적절한 CDM 방법을 매칭하여 에이전트의 의사결정 품질을 향상시킬 수 있는 가능성.
  - 사회 선택 이론은 집단적 선호를 다루므로, 언어 모델 정렬 및 집계와 같은 보다 폭넓은 학제간 NLP 연구에 영감을 줄 것으로 기대.

---

# Limitations

- **MCQA와 CDM의 불일치**  
  - MCQA가 집단 의사결정(CDM)과 완전히 일치하지 않음.  
  - LLM이 다중 선택 순위 작업에서 일관성이 부족함 (Zhao et al., 2024).

- **고정된 정답**  
  - 대부분의 MCQA 벤치마크는 미리 정해진 '정답'을 가지고 있음.  
  - CDM 과정은 절대적인 정답이 없는 상황에서도 관련성이 있음.  
  - 예를 들어, LLM의 편향 측정은 개별 에이전트의 '선호' 집계가 필요함.

- **미래 연구 방향 제안**  
  - 사실 여부 판단이 아닌 선호 대표성을 측정하는 벤치마크 구축 가능성.

- **자체 포함 테스트**  
  - 모든 실험은 단일 백본 모델의 자체 포함 시스템임.  
  - 서로 다른 LLM을 이용한 투표 에이전트 조합 테스트는 하지 않음.

- **GEDI에서 투표 전략 포함 부족**  
  - 현대 전자 시스템을 모두 포괄하지 않음.  
  - 여러 투표 전략을 결합한 복합 메커니즘은 제외됨.  
  - 필요 시 여러 GEDI 모듈을 연결하여 구현 가능.

- **'투표 세금'**  
  - 전자 CDM 방법의 '투표 세금'은 계산 비용을 의미함.  
  - 두 부분으로 구성: 에이전트 행동과 투표 처리.  
  - 에이전트 행동이 가장 큰 비율을 차지하며, LLM 운영이 매우 비용이 많이 듬.  
  - 에이전트 간 커뮤니케이션 비용도 고려해야 함.

- **참여의 비용-편익 균형**  
  - 인간 투표자는 결과와 관계없이 참여로 인해 만족감을 느낄 수 있음.  
  - 그러나 LLM 에이전트는 참여를 통해 이익을 얻지 못함.  
  - 이 차이로 인해 LLM 에이전트 CDM에서 투표 인구는 실용적인 요소로 작용함.  

- **광범위한 영향 미비**  
  - 추가적인 논의 필요.

---


# Ethical Considerations

- 본 연구의 목적은 LLM 기반 에이전트들 간의 다양한 집단 의사결정 방법을 탐구하는 것임.
- 연구는 LLM 에이전트를 인간의 판단을 대체하는 대표로 사용하는 행위를 지지하거나 권장하지 않음.


---

# A Reproducibility Statement

- 실험을 위해 8개의 백본 모델을 사용
- 상업적으로 이용 가능한 독점 모델:
  - gpt-3.5 
  - gpt-4 
  - 구체적으로, 스냅샷 모델인 gpt-3.5-turbo-1106 및 gpt-4-0125-preview 사용
- 오픈소스 모델:
  - Mistral-7B-v0.3
  - glm-4-9b-chat
  - Llama-3-8B/70B-Instruct
  - Qwen1.5-72B/110B-Instruct
- 위 모델들의 출처는 다음과 같음:
  - Mistral-7B: [mistral-7b 소스](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  - Llama-3-8B: [llama-3-8b 소스](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - glm-4-9b: [glm-4-9b 소스](https://huggingface.co/THUDM/glm-4-9b-chat)
  - Llama-3-70B: [llama-3-70b 소스](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
  - Qwen1.5-72B: [qwen1.5-72b 소스](https://huggingface.co/Qwen/Qwen1.5-72B-Chat)
  - Qwen1.5-110B: [qwen1.5-110b 소스](https://huggingface.co/Qwen/Qwen1.5-110B-Chat)
  - gpt-3.5: [gpt-3.5 소스](https://platform.openai.com/)
  - gpt-4: [gpt-4 소스](https://platform.openai.com/)

---

# B Surveyed LLM-based Multi-Agent Collaboration Frameworks and Systems

- **CDM Method Systems 및 Frameworks**
  - **독재적인 접근 방식**
    - Xiong et al. (2023): Assigned role
    - Wu et al. (2023): Assigned role
    - Hao et al. (2023): Assigned role
    - Liu et al. (2023b): Assigned role
    - Li et al. (2023a): Assigned role
    - Zhang et al. (2023a): Assigned role
    - Nair et al. (2023): Assigned role
    - Talebirad and Nadiri (2023): Assigned role
    - Liang et al. (2023): Assigned role
    - Tang et al. (2023): Assigned role
    - Qian et al. (2023): Assigned role
    - Sun et al. (2023): Assigned role
    - Chen et al. (2023a): Assigned role
    - Jinxin et al. (2023): Assigned role
    - Li et al. (2023b): Assigned role
    - Fang et al. (2024): Assigned role
    - Tang et al. (2024): Assigned role
    - Hang et al. (2024): Assigned role
    - D’Arcy et al. (2024): Assigned role
    - Hua et al. (2024): Assigned role
    - Wang et al. (2024b): Assigned role
    - Li et al. (2023f): Assigned role
    - Chen et al. (2023b): Oligarchy

  - **비 중앙집중식 팀**
    - He et al. (2023): Decentralized team
    - Li et al. (2023c): Decentralized team
    - Nakajima (2023): Decentralized team

  - **인간 판단**
    - Ni and Buehler (2024): Human judgement
    - Ghafarollahi and Buehler (2024): Human judgement

  - **선형 워크플로우**
    - Wang et al. (2023a): Linear workflow
    - Ding et al. (2023): Linear workflow
    - Hong et al. (2023): Linear workflow
    - Rasheed et al. (2024): Linear workflow
    - Wei et al. (2024): Linear workflow

  - **시나리오 시뮬레이션**
    - Liu et al. (2023a): Scenario simulation
    - Park et al. (2023): Scenario simulation
    - Ghaffarzadegan et al. (2023): Scenario simulation
    - Hua et al. (2023): Scenario simulation
    - Zhang et al. (2024): Scenario simulation

  - **다수결 및 합의**
    - Du et al. (2023): Consensus
    - Wang et al. (2023d): Consensus
    - Chen et al. (2023d): Consensus
    - Chen et al. (2023c): Consensus
    - Li et al. (2023d): Consensus
    - Shi et al. (2023): Game rule

  - **게임 규칙 및 상대 다수결**
    - Stepputtis et al. (2023): Game rule
    - Xu et al. (2023a): Game rule
    - Chan et al. (2023): Relative majority
    - Xu et al. (2023b): Relative majority
    - Zhang et al. (2023b): Relative majority
    - Li et al. (2024): Relative majority

  - **상황 시뮬레이션**
    - Hamilton (2023): Scenario simulation
    - Jarrett et al. (2023): Utilitarian

- **전체 목록**: 52개의 LLM 기반 다중 에이전트 협력 연구 작품의 목록.

---

# C Main Experiment Statistics

- MMLU 및 MMLU-Pro 데이터셋의 경우 주제별 균형 잡힌 테스트 하위 집합을 구성하여 각각 100개의 사례를 선택함.
  - MMLU: 5,700개 질문
  - MMLU-Pro: 1,400개 질문
- ARC-Challenge에서는 전체 1,172개의 테스트 세트를 사용.
- 유효한 프로필 조건:
  1. 모든 투표 에이전트의 투표가 포함되어야 함.
  2. 각 투표는 완전하고 중복되지 않은 순위 목록을 포함하며 지정된 형식에 맞춰야 함.
- 유효한 프로필만 GEDI에 전달되어 처리됨.
  
| 데이터셋       | 모델              | Range | Ordinal Ranking | Informed | Mis-informed |
|----------------|------------------|-------|---------|---------|----------|
| **MMLU**       | mistral-7b       | 2379  | 4788    | 5422    | 5596     |
|                | llama-3-8b      | 1253  | 1946    | 4961    | 5121     |
|                | glm-4-9b        | 332   | 3470    | 5502    | 5447     |
|                | llama-3-70b     | 3909  | 5110    | 5576    | 5435     |
|                | qwen1.5-72b     | 4642  | 5657    | 5698    | 5700     |
|                | qwen1.5-110b    | 5569  | 5625    | 5685    | 5692     |
|                | gpt-3.5-trubo   | 5627  | 5397    | 5569    | 5679     |
|                | gpt-4           | 5515  | 5572    | 5539    | 5648     |
| **MMLU-Pro**   | mistral-7b       | 554   | 564     | 1180    | 1382     |
|                | llama-3-8b      | 3(1161*)| 261   | 1162    | 1255     |
|                | glm-4-9b        | 3(1359*)| 376   | 1294    | 1323     |
|                | llama-3-70b     | 1239  | 1293    | 1396    | 1394     |
|                | qwen1.5-72b     | 388   | 831     | 1284    | 1383     |
|                | qwen1.5-110b    | 632   | 1138    | 1319    | 1399     |
|                | gpt-3.5-turbo   | 655   | 1283    | 1400    | 1400     |
|                | gpt-4           | 1375  | 1386    | 1399    | 1397     |
| **ARC-Challenge** | mistral-7b    | 373   | 1033    | 1131    | 1163     |
|                | llama-3-8b      | 252   | 317     | 1024    | 1043     |
|                | glm-4-9b        | 1(1096*)| 1081   | 1153    | 1159     |
|                | llama-3-70b     | 901   | 1135    | 1172    | 1172     |
|                | qwen1.5-72b     | 1068  | 1172    | 1172    | 1172     |
|                | qwen1.5-110b    | 1166  | 1169    | 1171    | 1171     |
|                | gpt-3.5-trubo   | 1172  | 1172    | 1172    | 1172     |
|                | gpt-4           | 1172  | 1172    | 1171    | 1172     |

- 유효한 프로필 수는 정보적 독재에서 필수 조건인 모든 비독재자의 투표 프로필이 필요하므로 원본 프로필 수보다 적음.
- *Llama-3-8b 및 glm-4-9b 모델은 특정 벤치마크에서 완전한 프로필 수가 너무 적어 유효한 투표가 포함된 불완전한 프로필을 사용하여 정확성을 계산.

---

# D Several CDM Method Criteria Examples

- **투표 시스템 예시**:
  - **다수결 투표의 부적절성**:
    - Amber가 초기 투표에서 가장 많은 1순위 표를 받아 승리.
    - Coral이 추가된 후, Amber와 Blue의 상대적 투표 위치는 유지되지만, Blue가 가장 많은 1순위 표를 얻음.
    - 예시: 
      - 초기: Amber (6) > Blue (4)
      - 이후: Blue (4) > Amber (3) = Coral (3)
  
- **Condorcet 기준 위반 예시**:
  - Blue가 다수결 승자이지만, Amber가 모든 쌍 비교에서 더 많은 선호 투표를 얻어 Condorcet 승자.
  - Amber의 1순위 표는 적지만, 쌍 비교에서 우위를 점함.

- **단조성 기준 위반**:
  - 즉시 runoff 투표(Instant-Runoff Voting)에서 반복적으로 가장 적은 1순위 표를 얻은 후보를 제거하여 승자를 결정.
  - 시나리오 1에서 Amber가 탈락, Coral이 승리.
  - 시나리오 2에서 유권자가 Coral을 1순위로 선택했지만, 결과적으로 Coral이 패배.

- **효용 기반 결정 방법의 문제**:
  - Blue는 유틸리티가 더 높은 승자 (U = 10×10 + 0×2 = 100).
  - Amber는 다수의 선호를 받아 Condorcet 승자.
  - Blue는 유틸리티에서 우위지만, Amber는 12명 중 10명이 선호. 

- **결론**:
  - 여러 투표 방법에서 다수 결정을 할 때, 선호 순위와 유틸리티를 고려해야 하며, 각 기준을 만족하는지 검토해야 함.