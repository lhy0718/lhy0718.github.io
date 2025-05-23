---
title: "[논문리뷰] Strength Lies in Differences! Improving Strategy Planning for Non-collaborative Dialogues via Diversified User Simulation (EMNLP 2024)"
date: 2025-04-07 17:01:00 +0900
categories:
  - Paper Review
tags:
  - EMNLP 2024
  - Persona-based Dialogue
---

이 논문에서는 시스템의 목표에 유리한 합의를 이끌어내기 위해 다양한 사용자와 전략적 대화를 수행하는 비협력적 대화 에이전트를 연구하고, 사용자 특성을 고려한 전략적 계획 및 일반화된 훈련을 위한 TRIP을 제안합니다. 실험을 통해 TRIP의 효과를 입증합니다.

---

# 1 Introduction

- 비협력 대화는 대리자와 사용자 간의 이해관계가 상충할 때 발생 (예: 협상, 설득).
- 양측은 자신에게 유리한 합의를 얻기 위해 다양한 전략을 사용해야 함.
- 사용자 저항은 대리자의 전략에 따라 달라지므로, 대리자는 다양한 사용자를 위해 전략 계획을 세워야 함.
- '일률적인' 전략에 의존하면 대리자가 적응성과 유연성이 결여되어 취약해질 수 있음.
- 최근 대규모 언어 모델(LLM)이 비협력 작업을 수행하는 대화 에이전트로 활용됨.
- 혼합 주도 프롬프트 또는 외부 전략 계획자를 통합하여 LLM의 응답을 안내하는 목표를 가짐.
- 그러나 이러한 시도는 실제 상황에서 성능에 대한 비판을 받고 있음.
  
  - 두 가지 주요 문제:
    1. 기존 방법은 대화 이력에만 의존하고 사용자 특정 특성을 전략 계획에 포함하지 않음. 개인 사용자에 대한 정보 기반 표현을 생성함으로써 대리자는 행동을 조정하고 맞춤형 전략을 수립할 수 있음.
    2. 훈련 패러다임이 다양한 사용자에게 일반화되는 전략 계획자를 생성하지 못함. 단일 사용자 시뮬레이터에 의존해 서로 다른 비협력 행동 생성을 제한함.
    
- 이러한 제한으로 인해 기존 LLM 기반 대화 에이전트가 다양한 사용자를 위한 전략 조정에 어려움을 겪고, 최적의 성과를 내지 못함.
  
- 평가 프로토콜을 설정하여 비협력 행동이 다양한 사용자 시뮬레이터를 배치하고, 현재 LLM 기반 대화 에이전트의 전략 계획의 한계를 조사.
- TRIP이라는 간단하면서도 효과적인 방법을 설계하여 LLM의 맞춤형 전략 계획 기능을 향상.
  
  - TRIP의 구성:
    - 사용자 인식 전략 계획 모듈: 사용자의 정신 상태와 미래 행동 분석.
    - 인구 기반 훈련 패러다임: 다양한 사용자 시뮬레이터로 훈련하여 전략 계획 모듈의 적응성 증대.
    
- 주요 기여:
  - 비협력 대화에서 다양한 사용자를 위한 전략 맞춤화의 중요성을 강조.
  - 사용자 인식 전략 계획 모듈과 인구 기반 훈련 패러다임을 포함하는 TRIP 제안.
  - 협상 및 설득과 같은 비협력 대화 작업에서 실험을 수행. TRIP이 다양한 사용자에 맞춘 전략을 사용하여 기본선보다 우수한 성과를 내는 것으로 나타남.

---

# 2 Related Work

- **연구 연결성**
  - 본 연구는 LLM 시대의 비협력적 작업을 해결하기 위한 전략적 계획 및 훈련 패러다임과 밀접하게 관련되어 있음.

- **비협력적 대화를 위한 전략적 계획**
  - 최근 연구들은 전략적 계획의 효과성을 높이기 위해 LLM 기반의 다양한 방법을 소개함.
  - **방법 유형 분류:**
    1. **자극 프롬프트 개발:**
       - (Chen et al., 2023)는 능동적 대화 문제를 해결하기 위해 혼합 주도 프롬프트의 효과를 검증함.
       - (Deng et al., 2023d)와 (Zhang et al., 2023a)는 LLM이 다음 행동을 계획할 수 있도록 자기 반성을 유도함.
       - (Fu et al., 2023)는 다른 LLM들로부터 피드백을 요청하여 전략적 계획을 반복적으로 개선하는 자기 플레이 시뮬레이션을 활용함.
       - 그러나 (Deng et al., 2023e)가 강조한 바와 같이, 비훈련 가능한 파라미터의 영향으로 이 방법들의 효과가 저해됨.
       
    2. **외부 전략 계획가 장착:**
       - 이 계획가는 매 턴마다 프롬프트를 생성하여 LLM에 더 섬세한 지침을 제공함.
       - 몬테 카를로 트리 탐색 방법 (Yu et al., 2023)이나 플러그인 모델 (Deng et al., 2023e)을 통한 통합이 가능하나, 사용자 특성을 전략 계획에 통합하는 데 어려움이 있음.
  
- **비협력적 대화를 위한 훈련 패러다임**
  - 현재 훈련 패러다임은 대화 에이전트가 단일 사용자 시뮬레이터와 상호작용하여 전략적 계획 능력을 향상시킴.
  - (Chawla et al., 2023b)는 감독 방식으로 인간-인간 대화 데이터를 모방하는 사용자 시뮬레이터를 구축함.
  - (Yu et al., 2023; Deng et al., 2023e)는 롤플레잉 LLM 기반의 사용자 시뮬레이터를 사용함.
  - 그러나 단일 사용자 시뮬레이터는 특정 사용자 행동만을 나타내기 때문에 다양한 사용자 행동을 반영하지 못할 수 있음.
  - 따라서 기존 훈련 패러다임은 다양한 행동을 가진 사용자에게 맞춤형 전략 계획가를 생성하지 못함.
  
- **본 연구의 기여**
  - 본 논문에서는 인구 기반 훈련을 통해 사용자의 행동을 다양화하는 맞춤형 전략 계획의 중요성을 조사함.

---

# 3 Strategic Planning Evaluation

- 새로운 평가 프로토콜 도입
  - 기존의 LLM 기반 대화 에이전트의 한계를 분석
  - 비협력적 행동을 보이는 사용자를 처리하지 못하는 문제 강조

- 전체 평가 과정 요약
  - 과정은 Figure 1에서 설명됨
  - 자세한 내용은 Appendix A 참조

- 평가 모델의 수식:
  - 성능 평가를 위한 수식: $$P = \frac{T_{success}}{T_{attempted}}$$
  - 비협력적 사용자의 행동을 평가하는 수식: $$B = \sum_{i=1}^{n} w_i \cdot \vert a_i - p_i \vert$$ (여기서 $a_i$는 실제 행동, $p_i$는 예상 행동, $w_i$는 가중치)

---

# 3.1 Evaluation Setup

- **평가 개요**
  - 다양한 비협력적 행동을 보이는 합성 사용자 시뮬레이터 환경 구성.
  - 각 대화 에이전트는 이 시뮬레이터와 상호작용.
  - 상호작용 동안 대화 에이전트와 사용자 시뮬레이터는 자신의 이익을 극대화하기 위해 응답 전략을 번갈아 사용.
  - 대화 목표 달성 또는 최대 턴 수 도달 시까지 상호작용 지속.
  - 상호작용 결과 수집 및 에이전트 성능 평가.

- **기준선**
  - 두 가지 기준선 고려:
    - **Standard agent**: 수정 없는 기본 LLM.
    - **PPDPP agent**: 최신 SOTA 에이전트로 훈련 가능한 외부 전략 계획기 포함.

- **비협력 사용자 시뮬레이터**
  - 비협력적 행동 기반으로 합성.
  - 비협력적 행동은 설계된 저항 전략을 통해 LLM에 명시적으로 주입.
  - 다양한 페르소나를 갖춘 LLM을 사용해 저항 전략 선택.
  - 두 가지 페르소나 유형: 
    - **Big Five Personality** 
    - **Decision-Making Styles**
  - 각 페르소나에 대한 일관된 설명 생성.

- **평가 작업**
  - 두 가지 비협력적 작업에서 실험 진행:
    - **가격 협상 작업**: CraigslistBargain (CB) 데이터셋 활용.
    - **자선 설득 작업**: PersuasionForGood (P4G) 데이터셋 활용.
  - 대화 에이전트는 각각 구매자 및 설득자 역할.

- **평가 메트릭**
  - 세 가지 일반적인 메트릭 고려:
    - **Success Rate (SR)**: 목표 달성 비율.
    - **Average Turn (AT)**: 목표 달성을 위한 평균 턴 수.
    - **Sale-to-List Ratio (SL%)**: 거래 완료의 효과성 측정.
  - SL% 공식: $$ \text{SL\%} = \frac{P_{\text{deal}} - P_{\text{seller target}}}{P_{\text{buyer target}} - P_{\text{seller target}}} $$
  - SL%가 높을수록 구매자에게 더 많은 이익. 거래 실패 시 SL%를 0으로 설정.

---

# 3.2 Experimental Findings

- 다양한 비협력적 행동을 가진 사용자 시뮬레이터에서 기존 대화 에이전트들의 성능을 분석.
- PPDPP와 Standard 에이전트를 비교하여 PPDPP의 성능 개선이 두드러지기는 하지만 다양한 비협력적 전략을 사용하는 사용자에게 잘 적응하지 못함.
- PPDPP는 17.77%의 경우에서 Standard보다 유의미한 개선을 보여주지 않으며 (예: 가격 협상에서 Analytical 사용자에 대해 SR을 0.02 증가).
- 8.88%의 경우에서는 Standard보다 성능이 저하됨 (예: 가격 협상에서 Neuroticism 사용자에게 SR을 0.02 감소).
- 이는 다양한 사용자에 맞춘 전략적 계획의 필요성을 부각시킴.

---

# 4 TRIP: Tailored Strategic Planning

- **목표**: TRIP은 사용자 특성을 추론하여 전략 계획 모듈에 통합.
- **기반 모델**: 학습 가능한 BERT를 활용.
- **이해 사용자 상태**:   
  - LLM의 Theory-of-Mind 능력을 활용하여 사용자의 mental states 및 future actions 예측.
  - 예를 들어, 사용자의 목표 가격이나 기부 의사를 이해함.
- **형식적 접근**: 
  - 대화 이력 $D = (usys_1, uusr_1, \ldots, usys_t, uusr_t)$를 LLM에 입력.
  - mental states $M$ 및 future actions $F$를 추론: $PLLM(M, F | D)$.
- **전략 예측**: 
  - {M, F, D}를 전략 계획자 $\pi_\theta$에 입력하여 다음 전략 예측.
  - 출력 공간은 사전 정의된 전략 집합.
  
---

# 4.1 User-Aware Strategic Planning

- **목적**: TRIP의 효과성을 평가하기 위해 Section 3.1의 평가 프로토콜을 따름.
  
- **전반적인 성과 보고**: 
  - Section 5.1에서 대화 에이전트의 전체 성과를 보고함.

- **심층 분석**: 
  - Section 5.2에서 TRIP의 맞춤 전략을 드러내기 위한 심층 분석 실시.

- **탈락 연구**: 
  - Section 5.3에서 다양한 사용자 인식 및 훈련 집단의 성능 변동을 정리하고, 맞춤 전략 계획을 위한 주요 예측 변수를 찾음.

- **LLM 기반 기준**:
  - 두 가지 유형의 전략 계획 모듈을 가진 LLM 기반 대화 에이전트를 고려:
    1. **프롬프트 기반 계획**: 
       - Standard, ProCoT (Deng et al., 2023d), ICL-AIF (Fu et al., 2023) 포함.
       - 혼합 주도 프롬프트, CoT, AI 피드백을 사용하여 다음 전략 선택.
    2. **외부 전략 계획기**: 
       - GDP-MCTS (Yu et al., 2023) 및 PPDPP (Deng et al., 2023e) 포함.
       - 몬테 카를로 트리 탐색 및 훈련 가능한 플러그인을 통해 다음 전략 결정.

- **사용자 특성 고려 부족**: 
  - 모든 기준은 사용자 특정 특성을 명시적으로 모델링하지 않으며, 하나의 사용자 시뮬레이터로 훈련됨.

- **성과 비교**: 
  - 다양한 페르소나에서의 에이전트 성능을 보고.
  - 가격 협상(왼쪽) 및 자선 설득(오른쪽) 작업에서 성공률 보고.
  - TRIP는 모든 페르소나에서 균형 잡힌 개선을 이루어 다른 에이전트보다 상당히 우수한 성과 제공.

- **평가 지표**: 
  - Section 3.1에서 언급된 자동 지표 사용.
  - 또한, 대화 에이전트의 실제 효과성을 평가하기 위한 인간 평가 실시.
  - 인간 평가에 대한 추가 세부 사항은 Appendix C에서 확인 가능.

---

# 4.2 Population-Based Training Paradigm

- **평가 방법**:
  - 모든 에이전트의 전체 및 세부 성과를 자동화된 메트릭을 사용하여 평가함.
  - 그림 4에서 실제 사용자와의 상호작용 시 성능을 인적 평가로 보고함.

- **TRIP의 성능**:
  - TRIP는 다양한 사용자를 위한 효과적인 비협력 전략을 달성하는 유망한 방법임.
  - 표 2에 따르면, TRIP는 두 가지 작업에서 모든 기준선에 비해 상당한 차이로 우수함.
    - 대화 목표를 효율적으로 달성 (AT↓ 감소)
    - 작업 성공률(SR)과 전략의 성공률(SL%) 높음.

- **사용자 페르소나에 대한 개선**:
  - 그림 3에서 TRIP는 다양한 사용자 페르소나에 대해 균형 잡힌 개선을 보임.
  - PPDPP의 편향된 개선과 대조적으로 다른 에이전트를 상당히 능가함.
  - 이는 TRIP가 다양한 사용자에게 잘 일반화되는 전략을 생성할 수 있음을 나타냄.

- **LLM 기반 사용자 시뮬레이터의 한계**:
  - 단일 LLM 기반 사용자 시뮬레이터의 행동 패턴은 범위가 제한적임.

- **인적 평가 결과**:
  - 그림 4에서 TRIP는 실제 사용자와의 상호작용에서 Standard 및 PPDPP를 크게 능가함.
  - PPDPP는 두 가지 작업에서 Standard 접근 방식을 일관되게 초과하지 않음.
    - 예: 협상 작업에서 더 높은 성공률을 달성하였으나, 더 많은 상호작용 라운드 필요.

- **주요 성과 데이터 (표 2)**:
  - 각 에이전트의 가격 협상 및 선의의 설득에서의 SR, AT, SL% 성과를 비교.
  - TRIP는 모든 기준선에 비해 가장 높은 SR과 낮은 AT를 기록함.

- **결론**:
  - TRIP는 실제 사용자와 효과적으로 상호작용할 수 있는 높은 실용성을 지님.

---

# 5 Experiments

- **목표**: TRIP의 맞춤형 전략 계획의 효과를 분석
- **전략 수집**: 사용자 상호작용에서 각 에이전트의 전략을 모아 전략 시퀀스를 형성
- **비교 방법**:
  - BERT와 t-SNE 사용하여 전략 시퀀스를 임베딩 벡터로 인코딩
  - 유클리드 거리 측정으로 같은 페르소나와 다른 페르소나 간의 평균 거리 계산
  - 메트릭: Intra-Persona (같은 페르소나간 거리), Inter-Persona (다른 페르소나간 거리)
  
- **실험 결과**:
  - TRIP는 가장 낮은 Intra-Persona와 가장 높은 Inter-Persona 달성
  - 이는 TRIP이 같은 페르소나의 사용자와 유사한 전략 시퀀스를 갖고, 다른 페르소나와는 뚜렷한 차이를 보임을 나타냄
  - TRIP은 인구 동태에 대한 인지가 더 뛰남

- **사례 연구(그림 5)**:
  - PPDPP는 다양한 사용자 유형에 대해 같은 전략 패턴을 반복하나, TRIP은 더 깊은 사용자 이해 기반으로 맞춤형 전략을 제공
  - Neuroticism 페르소나와의 상호작용에서 TRIP은 개인 경험과 논리를 활용
  - Openness 페르소나와의 상호작용에서 TRIP은 감정을 자극하도록 주장
  
- **전략 차이에 따른 효과**:
  - Openness 사용자는 감정에 의해 쉽게 영향을 받음
  - Neuroticism 사용자는 다른 사람의 개인 경험에 더 영향을 받음

- **주요 성과**:
  - TRIP은 비협력적 대화에 대한 향후 연구에 대한 통찰을 제공할 수 있는 전략적 차이를 보임
  
- **탐색 결과 (표 4)**:
  - 사용자 인식 전략 계획 모듈과 인구 기반 훈련이 에이전트를 개선하고 서로 보완함을 입증

---

# 5.1 Overall Performance

- **목적**: 사용자 인식 및 훈련 집단의 성능 변화를 분석.
  
- **모델 변형**:
  - **TRIP w/o POP**: 인구 기반 훈련 없이 고정된 LLM 기반 사용자 시뮬레이터와 훈련.
  - **TRIP w/o UA**: 사용자 인식 전략 계획 모듈 제거, 대화 기록만 사용.
  - **TRIP w/ 10 POP**: 10개의 페르소나를 이용한 인구 기반 훈련, 20개 페르소나 카테고리에서 무작위 선택.
  - **TRIP w/ 10 POP & w/o UA**: 사용자 인식 전략 계획 모듈을 제거한 10 POP 버전.

- **성능 요약**:
  - 사용자 인식 전략 계획 및 인구 기반 훈련은 맞춤형 전략 계획을 위해 효과적임.
  - **TRIP w/o UA**에서 **TRIP**가 설득 성공률은 $$0.3233 \rightarrow 0.4400$$, 거래 이익 SL%는 $$0.3144 \rightarrow 0.3505$$로 증가.
    - 사용자 정신 상태와 미래 행동을 포함하는 것이 효과적 전략 개발에 도움을 줌.
    - 그러나 거래 성공률은 $$0.6988 \rightarrow 0.6888$$로 약간 감소, 이는 사용자 특성을 깊게 모델링함이 거래 참여 의지를 감소시킬 수 있음.
  
- **다양한 사용자 시뮬레이터의 효과**:
  - **TRIP w/o POP**보다 SL%는 $$0.3505 \rightarrow 0.4096$$로 유의미하게 증가.
  - 다양한 훈련 집단은 대화 에이전트의 적응성을 향상시키나, 추가 훈련 도전 과제를 생성할 수 있음.
  
- **훈련 상호작용 성능 분석**:
  - 1000회 훈련 상호작용에서 TRIP w/o UA와 TRIP w/ 10 POP & w/o UA는 초기 400회에서 느린 수렴.
  - 훈련 사용자 시뮬레이터를 고정하지 않으면 초기 훈련 단계에서 불안정성을 유발할 수 있음.
  - 500회 이후 TRIP w/o UA의 훈련 과정이 안정화되며 다른 에이전트를 초월하는 성능 향상.
  - PPDPP는 특정 상호작용 후 성능 하락 관찰, 단일 사용자 시뮬레이터와의 광범위한 상호작용이 지속적인 성능 향상을 보장하지 않음.

---

# 5.2 Strategy Analysis

- 현재의 LLM 기반 대화 에이전트가 다양한 비협력 사용자에게 맞춤형으로 대응하지 못하는 문제를 조사함.
- TRIP 방법 제안:
  - 비협조적 대화에 대한 전략적 계획을 맞춤화하기 위해 설계됨.
  - 사용자 인식을 고려한 전략적 계획 모듈 포함.
  - 인구 기반 훈련 패러다임 사용.
- 실험 결과:
  - 다양한 사용자 그룹에서 TRIP의 효과성과 효율성을 입증.
- 연구의 기여:
  - 비협력적 대화 에이전트의 적응성과 유연성을 향상시키기 위한 기초 작업으로 간주됨.
- 향후 계획:
  - 초보 에이전트 훈련 및 코칭에 소요되는 자본 비용을 줄이기 위한 인구 인식 에이전트의 가능성 탐색.

---

# 5.3 Ablation Study

- **프롬프트의 민감도** 
  - LLM(대규모 언어 모델)에서 프롬프트의 영향을 분석 
  - 이전 연구(Deng et al., 2023d)와 유사하게, 평가 결과는 프롬프트에 의해 영향을 받을 것
  - 혼합 주도 형식을 활용하여 프롬프트를 설정, 안정성과 통제를 제공
  - 프롬프트의 영향과 최적성은 LLM 내에서 중요한 탐구 영역으로, 향후 연구에서 다뤄야 할 주제

- **제한된 비협력적 작업**
  - 두 가지 비협력적 대화 작업(가격 협상 및 자선 설득)만 실험 진행
  - 이 작업들은 고전적이고 널리 인정받는 벤치마크(Deng et al., 2023d; Chawla et al., 2023a)
  - 향후 연구에서는 제안된 TRIP를 더욱 다양한 비협력적 대화 시나리오에 적용할 계획 (Zhang et al., 2024; Zhou et al., 2023b; Zhang et al., 2023b)