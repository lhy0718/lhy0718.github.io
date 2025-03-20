---
title: "[논문리뷰] Towards Personalized Dialogue Without Pre-defined Profiles through In-Dialogue Learning (EMNLP 2024)"
date: 2025-03-19 00:00:00 +0900
categories:
  - Paper Review
tags:
  - ArXiv
  - Persona-based Dialogue
---

요약: 이 논문은 사전 정의된 프로필 없이 대화 내에서 페르소나 정보를 학습하는 IDL(In-Dialogue Learning) 프레임워크를 제안하며, 이를 통해 BLEU와 ROUGE 점수가 각각 최대 200%, 247% 향상됨을 보여줌.

---

# 1 Introduction

<img width="437" alt="image" src="https://github.com/user-attachments/assets/edc30ef0-6fe0-4734-9d6b-ef32921204aa" />

- 개인화된 대화 시스템(personalized dialogue systems)은 다양한 페르소나(persona)에 맞는 응답을 생성하는 기술로 주목받고 있음.
- 기존 접근법들은 사전 정의된 프로필(pre-defined profiles)에 의존하여 페르소나 정보를 제공하지만, 이는:
  - 시간과 비용이 많이 듦.
  - 유연성이 부족함.
- **In-Dialogue Learning (IDL)**을 제안하여 사전 프로필 없이 대화 내에서(persona learning from dialogue history) 페르소나 정보를 학습하는 방식을 도입함.
- IDL을 통해 기존 방법 대비 BLEU와 ROUGE 점수가 각각 최대 **200%**, **247%** 향상됨.

## 기존 접근 방식과 한계
- 기존 개인화 대화 시스템은 대부분 **사전 정의된 프로필**을 기반으로 페르소나를 설정함.
- 문제점:
  - 프로필 생성이 번거롭고 비용이 많이 듦.
  - 정적인 정보만 포함되어 있어 새로운 정보를 반영하기 어려움.
- IDL은 사전 프로필 없이 **대화 내 정보만으로 페르소나를 학습**할 수 있는 새로운 프레임워크를 제안함.

## IDL 개요
- **Mutual Supervised Learning (MSL)**: 대화 내에서 페르소나 정보를 학습.
  - `Static Persona Identification (SPI)`: 대화 내 페르소나 관련 정보를 클러스터링.
  - `Dynamic Persona Identification (DPI)`: 대화 흐름을 유지하면서 적절한 순서로 재정렬.
- **Deep Personalized Alignment (DPA)**: 대화 모델의 응답을 페르소나에 맞게 정교화.
  - `Direct Preference Optimization with Criterion (DPOC)`: 기존 DPO 기반 최적화 기법을 확장하여 더 정교한 페르소나 정렬 수행.

## 실험 결과
- IDL은 사전 프로필 없이도 기존의 **프로필 기반(persona-based) 대화 모델과 유사한 성능**을 달성함.
- IDL 적용 시 BLEU 및 ROUGE 점수가 대폭 상승하며, **기존 In-Context Learning (ICL)** 방법보다 뛰어난 성능을 보임.

## 주요 기여
1. **IDL 프레임워크 제안**: 사전 프로필 없이 대화 내에서 페르소나를 학습하는 최초의 방법론.
2. **정적 및 동적 페르소나 식별 기법 제안**: 페르소나 정보의 효율적 학습을 위한 데이터 조직화 방법 개발.
3. **DPOC 기반 최적화 도입**: 모델이 페르소나 정보를 보다 정밀하게 반영하도록 유도.
4. **다양한 데이터셋에서 검증**: ConvAI2, Cornell Movie-Dialogs Corpus, LIGHT 등의 데이터셋에서 IDL이 높은 성능을 보임.

---

# 2 Related Work

## 2.1 Personalized Dialogue Systems
- 개인화된 대화 시스템 연구는 페르소나 정보를 활용하는 방식에 따라 **세 가지 유형**으로 분류됨:
  1. **구조화된 데이터베이스 기반**: 표(table)와 같은 구조화된 데이터 활용
     - 한계: 데이터 부족으로 인해 응답 다양성이 제한됨.
  2. **텍스트 프로필 기반**: 사용자의 프로필을 텍스트 형태로 모델에 제공
     - 한계: 프로필 생성이 번거롭고, 사용자의 성격을 완전히 반영하기 어려움.
  3. **대화 세션에서 페르소나 정보 추출**: 대화 데이터를 통해 페르소나를 학습
     - **DHAP (Ma et al., 2021)**: 트랜스포머 기반 접근법 사용하지만, 대화 상대의 발화를 고려하지 않아 페르소나 정보 손실 발생.
     - **MSP (Zhong et al., 2022)**: 유사한 대화를 검색하여 페르소나 정보를 수집하지만, 일부 토큰만 선택하는 방식으로 일관성이 부족함.
     - **IDL(본 논문 기법)**: 대규모 언어 모델을 활용하여 프로필 없이 페르소나 정보를 효과적으로 학습하고, 성능을 향상시킴.

## 2.2 In-Context Learning (ICL)
- 최근 대규모 언어 모델(LLM)의 발전과 함께 **In-Context Learning (ICL)** 기법이 주목받음.
- **ICL의 주요 특징**:
  - **문맥 내 학습(demonstrations within context)**을 통해 모델이 특정 작업을 수행하도록 유도.
  - 사전 학습된 데이터가 아닌, 입력 예시(context)를 기반으로 즉시 학습하는 방식.
- **ICL의 한계**:
  - 일반적인 언어 모델 학습 목적이 ICL을 위해 설계되지 않았기 때문에 성능이 제한적임.
  - 문맥 내 샘플 선택 및 배치 방식이 결과에 큰 영향을 미침 (Zhao et al., 2021; Lu et al., 2021).
- **IDL과 ICL의 차이점**:
  - ICL은 개별 샘플을 학습하는 방식이지만, **IDL은 대화 흐름을 고려한 다중 발화(multi-turn) 기반 학습**을 수행함.
  - IDL은 강화학습 기법을 도입하여 페르소나 기반 대화를 더 효과적으로 생성할 수 있도록 설계됨.

---

# 3 Method

<img width="887" alt="image" src="https://github.com/user-attachments/assets/62c594e3-4ea5-48c3-ac68-9867555d2643" />

## 3.1 문제 정의 (Problem Formalization)
- **목표**: 사용자의 대화 기록 $$D_u$$ 를 기반으로 개인화된 응답을 생성하는 것.
- **입력**: 사용자 $$u$$ 와 상대방 $$v$$ 간의 대화 $$d_{(u,v)} = (q_1, r_1, \dots, q_t, r_t) \in D_u$$
  - 여기서 $$q_i$$ 는 상대방의 질문, $$r_i$$ 는 사용자의 응답.
- **모델의 학습 목표**:
  $$
  r_i = LM_\Theta (C_i, D_u)
  $$
  - $$LM$$ 은 언어 모델, $$\Theta$$ 는 학습 가능한 파라미터.
  - $$C_i = (q_1, r_1, \dots, q_i)$$ 는 현재 대화 컨텍스트.

## 3.2 상호 감독 학습 (Mutual Supervised Learning, MSL)
- **개념**: 이전 대화 내 페르소나 정보를 학습하여 개인화된 응답을 생성하는 것.
- **문제점**:
  1. **불필요한 정보 혼합**: 관련 없는 대화 내용이 모델 학습을 방해함.
  2. **비일관된 대화 흐름**: 여러 대화의 부적절한 연결이 자연스러운 대화를 방해함.
- **해결 방법**:
  - **정적 페르소나 식별 (Static Persona Identification, SPI)**:
    - 대화 데이터를 **페르소나 관련성** 기준으로 클러스터링.
  - **동적 페르소나 식별 (Dynamic Persona Identification, DPI)**:
    - 클러스터 내 대화 순서를 **대화 편집 거리 (Conversation Edit Distance, convED)** 를 활용하여 재정렬.
  - **convED 계산**:
    $$
    lev(i, j) = \min
    \begin{cases}
      lev_{ins}(i, j) \\
      lev_{del}(i, j) \\
      lev_{sub}(i, j)
    \end{cases}
    $$
    - $$lev_{ins}(i, j) = lev(i, j-1) + 1$$ (삽입 비용)
    - $$lev_{del}(i, j) = lev(i-1, j) + 1$$ (삭제 비용)
    - $$lev_{sub}(i, j) = lev(i-1, j-1) + \lambda \cdot w_{sub}(A_i, B_j)$$ (대체 비용)

## 3.3 심층 개인화 정렬 (Deep Personalized Alignment, DPA)
- MSL 이후에도 LLM이 정확한 페르소나 기반 응답을 생성하는 데 부족함이 있음.
- 해결책: **기준을 활용한 직접 선호 최적화 (Direct Preference Optimization with Criterion, DPOC)** 적용.

### 3.3.1 DPOC (Direct Preference Optimization with Criterion)
- 기존 DPO(Direct Preference Optimization) 방법의 문제:
  - 선택된 응답과 거부된 응답 간의 확률 차이를 최대화하는 것이 목표.
  - 하지만, 선택된 응답의 보상이 감소하면 효과가 떨어지는 문제(선호 저하, preference degradation) 발생.
- 해결책: **패널티 항목 추가**:
  $$
  P(r_w, r_l) = -\min(0, \log r_w - \log r_l)
  $$
  - $$r_w$$: 더 나은 샘플의 보상
  - $$r_l$$: 더 낮은 샘플의 보상
  - 보통 $$r_w > r_l$$ 일 때는 $$P(r_w, r_l) = 0$$, 하지만 $$r_l > r_w$$ 일 때는 패널티 적용.

- **DPOC 손실 함수**:
  $$
  L_{DPOC}(r_{cho}, r_{rej}, r_{crt}) = L_{DPO}(r_{cho}, r_{rej}) + P(r_{cho}, r_{crt}) + P(r_{crt}, r_{rej})
  $$
  - $$r_{cho}$$: 선택된 샘플의 보상
  - $$r_{rej}$$: 거부된 샘플의 보상
  - $$r_{crt}$$: 기준 샘플의 보상 (중간값 역할)

### 3.3.2 데이터 구성 (Data Construction)
- DPOC 적용을 위해 기준 샘플을 생성해야 함.
- 세 가지 주요 기준 샘플 유형:
  1. **일관성 부족 (Inconsistency)**:
     - 대화 컨텍스트에서 설정된 페르소나와 충돌하는 정보 포함.
  2. **허구적 정보 (Fabrication)**:
     - 대화에서 언급되지 않은 새로운 성격 정보를 추가.
  3. **반전 (Inversion)**:
     - 상대방의 페르소나 정보를 잘못 반영.
- 기준 샘플 생성 과정:
  - 기존 대화 데이터 $$D_u$$ 에서 페르소나 관련 정보를 추출.
  - 잘못된 정보를 삽입하여 $$h_{crt}$$ 생성.

---

# 4 Experiments

## 4.1 Datasets
- **ConvAI2**: 개인화된 대화를 위한 고품질 영어 데이터셋.
- **Cornell Movie-Dialogs Corpus**: 220,000개 이상의 영화 대화 포함.
- **LIGHT**: 판타지 기반 텍스트 어드벤처 게임에서 수집한 대화 데이터.

## 4.2 Baselines
### **프로필 기반(Profile-based) 방법**
- **GPT-2**: 텍스트 생성 능력이 뛰어난 모델.
- **PerCVAE**: Conditional Variational Autoencoder를 활용하여 페르소나 정보를 조건으로 반영.
- **BoB**: BERT 기반의 일관성 평가 및 생성 방식을 결합한 모델.
- **CLV**: 페르소나 정보를 그룹화하여 대화 일관성을 높이는 방법.

### **프로필 없이(Profile-free) 학습하는 방법**
- **DHAP**: 대화 이력을 활용하여 개인화된 응답을 생성.
- **MSP**: 과거 대화를 검색하여 유사한 대화를 활용하는 방식.

### **대형 언어 모델(LLM) 기반 접근법**
- **LLaMA-2-7B/13B IDL**: ConvAI2 데이터셋을 활용하여 IDL을 적용한 LLaMA-2 기반 모델.
- **LLaMA-2-7B/13B Gold**: 프로필 정보를 사용하여 미세 조정된 모델.
- **LLaMA-2 System**: 프로필 정보를 직접 제공하여 대화 생성.
- **LLaMA-2 FT**: ConvAI2에서 모든 대화를 개별 예제로 사용하여 미세 조정한 모델.

## 4.3 Evaluation Metrics
- **BLEU**: 대화의 일관성을 평가.
- **ROUGE-L**: 대화의 일관성과 관련성을 평가.
- **Distinct-1/2**: 생성된 응답의 다양성을 측정.
- **P-F1 & P-Co(Persona Cosine Similarity)**: 페르소나 일관성 평가.
- **Con.Score & Coh-Con.Score**: 모델의 응답이 주어진 페르소나와 얼마나 일치하는지를 평가.

## 4.4 Main Results
### **4.4.1 자동 평가(Automatic Evaluation)**

<img width="887" alt="image" src="https://github.com/user-attachments/assets/4bc09f66-9b96-4e3a-89ac-7d143404afa9" />

- **ConvAI2 데이터셋 결과**:
  - IDL을 적용한 LLaMA-2 모델이 기존 방법보다 **모든 평가 지표에서 성능이 우수**함.
  - IDL이 **대화 내에서 페르소나 정보를 효과적으로 학습**할 수 있음을 증명.
  - 프로필 없이도 프로필 기반 모델(LLaMA-2 Gold)과 유사한 성능을 달성.

<img width="887" alt="image" src="https://github.com/user-attachments/assets/ec82f87f-356b-4e51-bda9-574e2f65ffb4" />

- **Movie 및 LIGHT 데이터셋 결과**:
  - **ICL보다 IDL이 훨씬 우수한 성능을 보임**.
  - ICL은 단순한 텍스트 구조를 학습하는 데 집중하지만, IDL은 **대화 내에서 페르소나 정보를 활용하는 데 효과적**임.

<img width="449" alt="image" src="https://github.com/user-attachments/assets/ce37dace-c769-4e1d-b19d-bba379e07124" />

<img width="449" alt="image" src="https://github.com/user-attachments/assets/faba5aee-b208-49ab-9848-feb2f8e61862" />

## 4.5 Ablation Study (세부 분석)
- **기준 샘플(Criterion Sample)이 중요한 역할**을 함.
- **DPOC(DPO with Criterion) 기법이 페르소나 정보 습득을 강화**하며, 제거 시 성능 저하.
- **정적/동적 페르소나 식별(SPI/DPI) 기법이 훈련 데이터 정리를 돕는 역할**.

## 4.6 대화 세션 수의 영향
- 대화 세션 수가 증가함에 따라 IDL과 ICL 모두 성능 향상을 보임.
- **IDL이 더 빠르게 성능이 향상**되며, **대화 내에서 지속적인 학습이 가능함을 입증**.

---

# 5 Conclusion

## 연구 개요
- 본 연구에서는 **In-Dialogue Learning (IDL)** 프레임워크를 제안하여, 사전 정의된 프로필 없이 대화 내에서 페르소나 정보를 학습하는 방법을 소개함.
- 기존 접근 방식과 달리, IDL은 대화 데이터를 활용하여 **대규모 언어 모델(LLM)**에 적용 가능함.
- 자동 평가 및 인간 평가를 통해 IDL의 개인화 응답 생성 성능이 검증됨.

## 연구 한계 (Limitations)
1. **실험 모델의 한정성**  
   - 연구는 **LLaMA-2** 계열 모델에 집중하여 실험이 진행되었음.  
   - 따라서, 다른 대형 언어 모델(LLM)에서도 같은 성능을 보장할 수 없음.
  
2. **복잡한 페르소나 특성 처리의 어려움**  
   - IDL이 **상충되거나 변화하는 페르소나 특성**을 처리하는 능력은 검토되지 않음.  
   - 일관성이 부족한 사용자 정체성을 다룰 때 한계가 있을 수 있음.

3. **데이터셋의 제약**  
   - 연구에 사용된 데이터셋(ConvAI2, LIGHT 등)은 대화 내에서 **일관된 페르소나 정보**를 포함하고 있음.  
   - 실제 환경에서는 이러한 조건이 항상 유지되지 않을 가능성이 있음.

## 윤리적 고려 (Ethics Statement)
- 대화 및 페르소나 정보는 **개인의 민감한 정보**를 포함할 가능성이 있음.  
- 연구에서 사용된 데이터셋은 연구 범위 내에서 제한적으로 활용되었으며, **개인정보를 포함하지 않도록 조치**함.  
- 사용된 데이터셋은 **공개적으로 이용 가능한 데이터**이며, 사용된 모델은 **해당 라이선스를 준수**하여 학문적 및 윤리적 기준을 충족함.

---

# 독자 의견

- 본 논문은 **프로필 없이 대화 내에서 페르소나 정보를 학습**하는 IDL 프레임워크를 제안함.
- 이를 통해 **대규모 언어 모델(LLM)을 활용하여 페르소나 정보를 효과적으로 학습**하고, 성능을 향상시킴.
- 실시간 대화에서 학습에 대한 보틀넥을 줄일 필요가 있음
