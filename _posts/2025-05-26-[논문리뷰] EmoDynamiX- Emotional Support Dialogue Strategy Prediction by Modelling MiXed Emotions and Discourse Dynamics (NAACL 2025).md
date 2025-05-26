---
title: "[논문리뷰] EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics (NAACL 2025)"
date: 2025-05-26 15:07:59 +0900
categories:
  - Paper Review
tags:
  - NAACL 2025
  - Decision Making
  - Empathic Dialogue Systems
---

본 논문은 정서적 지원 대화 시스템의 투명성과 성능 향상을 위해 전략 예측과 언어 생성을 분리하고, 사용자 감정과 시스템 전략 간 상호작용을 그래프로 모델링하는 EmoDynamiX 프레임워크를 제안한다.

---

# 1 Introduction

- 인생의 어려움으로 고통받는 이들에게 조기 개입을 제공하는 것은 긍정적인 생활 방식으로 전환할 수 있도록 돕고, 더욱 배려하는 사회를 만드는 데 필수적임.  
- 이러한 필요성으로 인해 NLP 커뮤니티에서는 효과적인 감정 지원 대화(ESC) 시스템 개발에 주력해 왔음 (Liu et al., 2021).  
- ESC 시스템은 도움을 구하는 이들의 고통 완화를 목적으로 하며, 의료 전문가를 찾도록 돕는 첫 단계로 볼 수 있음.  
- 최근 다중 턴과 인간 평가가 포함된 ESC 데이터셋이 공개되면서 데이터 기반 접근법이 규칙 기반 방법을 능가하기 시작함 (Van der Zwaan et al., 2012).  
- 이전 연구는 주로 Clavel et al. (2022)가 정의한 모듈식 대화 시스템에 초점을 맞춤: 인식(recognition), 계획(planning), 생성(generation)의 3단계 워크플로우를 포함.  
- 예시로는 Tu et al. (2022), Deng et al. (2023), Liu et al. (2021), Cheng et al. (2022) 등이 있음.  
- 이러한 시스템에서는 사용자 상태 인식에 기반해 사회-정서 전략이 선택되고, 예측된 전략에 맞춘 맞춤형 언어 디코더를 통해 응답 생성함.  
- 대형 언어 모델(LLM)의 등장으로 향상된 기능이 제공되면서, 특히 엔드 투 엔드 ESC 시스템에서 LLM이 점점 주도적 역할을 하게 됨 (Chen et al., 2023b; Zheng et al., 2023).  
- 그러나 LLM을 이용한 암묵적 대화 전략 계획에는 두 가지 문제점이 있음:  
  1. LLM의 "블랙박스" 특성으로 인해 의사결정 과정의 투명성 부족 (Ludan et al., 2023; Chhun et al., 2024; Lu et al., 2024).  
  2. 사전 학습 데이터에서 유래한 선호 편향으로 인해 사회적 목표와 과제 지향 목표 간 균형 유지에 어려움이 있음.  
- 예를 들어, Abulimiti et al. (2023)은 ChatGPT가 낮은 친밀감 회복에 적합한 헤징 전략을 덜 사용하는 등 부적절한 전략을 선호함을 발견.  
- Kang et al. (2024)은 LLM의 특정 전략에 대한 강한 편향이 ESC 단계 결과를 저해할 수 있다고 보고함.  
- 이에 대한 해결책으로 외부 전략 플래너 도입이 유망함: 특정 맥락에서 부적절한 전략을 명시적으로 배제하여 통제력을 높임.  
- 명시적 의사결정 모듈은 편향 완화 및 대화 전략 숙련도를 높여, 전반적인 생성 품질을 향상시키는 것으로 평가됨 (Kang et al., 2024).  
- 이 배경 하에 본 연구는 중간 단계였던 대화 전략 예측에 집중하며 세 가지 목표를 설정:  
  1. 인간 전문가 전략과의 높은 정렬성  
  2. 선호 편향 감소  
  3. 투명성 향상  
- 또한, 점점 발전하는 LLM과 제어된 생성 기술(Dathathri et al., 2020) 덕분에 대화 전략 예측은 경제적인 문제 해결 경로로 각광받음.  
- 명시적 대화 전략 예측은 최신 LLM과 호환 가능한 플러그 앤 플레이 모듈이나, 향후 강화학습 기반 방법의 기초 구성 요소로 활용 가능하며 (Deng et al., 2024), 인간 전문가 전략과의 정렬성이 핵심임.  
- 본 연구는 사회-정서 전략 예측을 독립 과제로 다루며, 해당 과제를 Figure 1에 시각화함.  

**연구 질문 (RQ)**  
- **RQ1:** 투명성을 기본 설계로 갖추면서, LLM 프롬프트나 파인튜닝보다 우수한 전략 예측 성능을 가진 전용 프레임워크 구축이 가능한가?  
- **RQ2:** 감정 인식 대화 모듈(ERC)을 활용하여 ESC에서 사용자 감정을 반영함으로써 전략 예측 성능을 향상시킬 수 있는가?  

**본 연구 주요 기여**  
- RQ1 해결을 위해 EmoDynamiX라는 의사결정 프레임워크 제안:  
  - 여러 전문가 모델 결합  
  - 이종 그래프 학습 모듈로 전략과 사용자 감정 간 동적 상호작용 포착  
  - 그래프로 의사결정 과정을 추적하여 투명성 강화  
  - 역할 인지형 정보 집계용 더미 노드 사용으로 성능 향상 (Liu et al., 2022; Scarselli et al., 2008)  
- RQ2 해결을 위해 혼합 감정 모듈 설계:  
  1. 이산 레이블 대신 감정 분포 사용으로 ERC와 ESC 데이터셋 간 도메인 간극으로 인한 오류 전파 감소  
  2. 감정 분포 조정으로 1차 감정들 융합해 미묘한 감정 범주 효과적 모델링  
- 제안 프레임워크는 두 공용 ESC 데이터셋에서 기존 기법을 뛰어넘는 F1 점수 향상과 편향 감소를 입증함.

---

# 2 Related Works

- **2.1 Emotional Support Conversation (ESC)**
  - ESC의 목표는 사회적 지향(social-oriented)과 과제 지향(task-oriented) 모두에 해당하며, 공감을 표현하고 제안을 제공하여 고통 완화를 목표로 함 (Liu et al., 2021; Cheng et al., 2022).
  - 사용자 상태 모델링은 ESC에서 중요한 주제임.
  - 기존 연구들은 주로 상식 지식 그래프를 쿼리하는 방식을 사용 (Tu et al., 2022; Deng et al., 2023; Peng et al., 2022; Zhao et al., 2023; Li et al., 2024).
    - 쿼리는 현재 발화와 특정 지식 관계(예: xReact)를 이어붙여 구성됨.
    - COMET(상식 지식 그래프에 사전학습된 생성 모델)을 통해 상황에 대한 사용자의 감정 반응(xReact)을 반환.
  - 그러나 상식 지식은 세밀한 감정 상태를 포착하기엔 너무 일반적임.
  - 문맥 인지 아키텍처(순차적, 그래프 기반 등)로 대화 전문 모델은 미묘한 감정을 더 효과적으로 처리 가능.
  - 실제 상황에서는 감정이 혼합되어 나타나며, 상반된 감정(예: 슬픔과 기쁨)이 공존할 수 있음 (Braniecka et al., 2014).
  - 본 연구의 혼합 감정 모델링 접근법은 이러한 복잡성을 더 잘 처리하며, 기본 감정을 조합해 많은 세밀한 감정 표현을 추가적인 인적 주석 없이 모델링 가능(6절 참조).
  - EmoDynamiX의 핵심 차별점:
    1. 지식 기반 사용자 상태 모델링에 대한 대안으로, 사전 학습된 ERC 모델이 예측한 레이블 분포에 기반한 혼합 감정 모듈 제공.
    2. 이전 연구들이 다양한 대화 그래프 구조를 탐구한 반면(Li et al., 2024; Peng et al., 2022; Zhao et al., 2023), 본 연구는 다양한 대화 업무에서 효과적인 담화 구조를 ESC에 도입(Chen and Yang, 2021; Li et al., 2023; Zhang et al., 2023).

- **2.2 Graph Learning in Conversational Tasks**
  - 그래프 기반 접근법은 대화 관련 여러 작업에서 효과적임.
  - 감정 인식과 대화 행위 인식 같은 인식 작업에서는, 대상 화자 발화가 그래프 구조에 따라 인접한 발화들로부터 정보를 집계함.
  - Ghosal et al. (2019), Ishiwatari et al. (2020), Wang et al. (2020), Fu et al. (2023), Shen et al. (2021) 등은 화자 역할 간 상호작용을 기반으로 대화 그래프를 설계함.
  - Li et al. (2023), Zhang et al. (2023)은 전문가 모델로 분석한 담화 의존성에 기반해 대화 그래프를 구성하며, Chen and Yang (2021), Feng et al. (2021)은 이를 대화 요약에 적용.
  - Yang et al. (2023)은 상식 지식을 이종 노드로 통합함.
  - Hu et al. (2021b), Chen et al. (2023a)은 대화 그래프에서 다중 모달 융합을 모델링함.
  - 예측 대화 작업(예: 다음 대화 행위 예측)에서는 그래프에서 추출된 전역 정보를 활용.
    - 기존 연구들은 평균/최대 풀링(Joshi et al., 2021), 선형 계층(Raut et al., 2023) 같은 간단한 readout 함수를 사용.
  - 본 연구는 정보 집계를 위한 특수 자리 표시자(dummy nodes)를 그래프에 도입.
    - dummy nodes는 그래프 분류, 부분 그래프 동형 매칭(Liu et al., 2022) 등 다른 그래프 학습 작업에서 readout 함수 대안으로 사용되었으나,
    - 예측 대화 작업에 dummy nodes를 사용하는 것은 본 연구가 최초이며, 이전 화자 발화와 역할 인지 상호작용을 명확히 모델링하는 데 특히 유용함.

---

# 3 Problem Formulation

- 다음 대화 전략을 예측하는 작업은 다중 클래스 분류 문제로 표현할 수 있음.
- 대화가 T개의 화자 턴으로 구성되어 있다고 가정.
- 대화 이력을 $$H_T = \{U_T, A_T, S_T\}$$로 정의함.
  - $$U_T = \{u_t\}_{t=1}^T$$는 발화 발화들의 시퀀스로, 각 $$u_t = \{w_n\}_{n=1}^{N_t}$$는 $$N_t$$개의 단어로 이루어진 시퀀스임.
  - $$A_T = \{a_t\}_{t=1}^T$$는 화자 역할의 시퀀스로, $$a_t \in \{\text{user, system}\}$$임.
  - $$S_T = \{s_t\}_{t=1}^T$$는 가능한 전략들의 시퀀스임. 다만 전략은 에이전트(시스템)에게만 존재함.
- 전략 집합을 $$S$$라고 표기.
- 사용자 발화 턴의 인덱스 집합을 $$I_{\text{user}} = \{t \mid a_t = \text{user}\}$$, 에이전트 발화 턴 인덱스는 $$I_{\text{agent}}$$로 정의.
- $$t \in I_{\text{agent}}$$일 때 $$s_t \in S$$이고, $$t \in I_{\text{user}}$$일 때 $$s_t = \emptyset$$임.
- 고정된 윈도우 크기 $$N-1$$가 주어졌을 때, 에이전트의 다음 전략 $$s_{t=N}$$을 예측하는 것이 과제임.
- 이를 확률 분포 $$P(s_t \vert H_{1}^{t-1}) \quad \text{where} \quad s_t \in S, t \in I_{\text{agent}}$$ 형태로 모델링함.
- 이후 문맥 윈도우 기준으로 인덱싱을 재정의하여 전체 대화가 아닌 윈도우 내에서만 인덱스를 사용함.

---

# 4 Methodology

- 본 프레임워크는 세 가지 주요 구성요소로 이루어짐:
  1. 대화 맥락의 의미를 포착하는 **의미 모델링 모듈**
  2. 대화 이력에서 사용자의 감정과 시스템 전략 간 복잡한 상호작용을 포착하는 **이종 그래프 학습 모듈**
  3. 앞선 모듈들의 특징을 통합하여 예측 결과를 산출하는 **MLP 분류 헤드**

---

## 4.1 Semantic Modelling

- 대화 이력의 전역 의미 정보를 효과적으로 파악하기 위해, 대화 맥락을 평탄화된 시퀀스 형식으로 표현:

  $$
  <context> = [a_1], u_1, [a_2], u_2, \ldots
  $$
  
- 각 발화(turn)마다 화자 역할을 구분 토큰으로 사용하여 시작점을 표시
- RoBERTa 인코더에 입력:
  $$
  C = \text{RoBERTa}([CLS], <context>)
  $$
- 마지막 은닉층의 [CLS] 토큰 임베딩 $$C[CLS]$$ 을 대화 이력의 전역 의미 표현으로 활용

---

## 4.2 Heterogeneous Graph Learning

- 대화 이력의 상호작용 역학을 모델링하기 위해 이종 그래프 (heterogeneous graph, $$G = \{V, B\}$$) 사용
  - $$V$$: 노드 집합, $$B$$: 엣지 집합
- 노드 유형:
  - 사용자 감정 노드: 사용자 턴 $$i \in I_{user}$$ 의 미세 감정 상태를 표현
  - 시스템 전략 노드: 에이전트 턴 $$i \in I_{agent}$$ 의 대화 전략 정보 표현
  - 더미 노드 $$v_N$$: 예측 대상 발화를 위한 자리 표시자, 다른 노드 및 상호작용에서 정보 집계
- 엣지 유형:
  - 담화 의존성 (Discourse dependencies): 감정 및 전략 노드 간, 내부 연결 (Asher et al. (2016) 정의에 따라 여러 유형 존재, STAC 데이터셋 기반 사전학습된 담화 파서를 사용)
    - 모든 $$(i, j)$$ 에 대해 $$\langle v_i, v_j \rangle \in R_{Discourse}$$
  - 자기참조 엣지 (self-reference): 모든 시스템 전략 노드에서 더미 노드로 $$\langle v_i, v_N \rangle = r_{self}$$ 
  - 상호참조 엣지 (interreference): 모든 사용자 감정 노드에서 더미 노드로 $$\langle v_i, v_N \rangle = r_{inter}$$
- 엣지 집합: $$R = R_{Discourse} \cup \{r_{self}, r_{inter}\}$$

---

### 4.2.1 User Emotion Node Embedding: Mixed Emotion Method

- 미리 학습된 ERC(감정 인식 대화 모델)를 활용하여 사용자 발화의 감정 분포 예측
- 감정 분포에서 얻은 정보를 혼합 프로토타입 접근법으로 사용자 세밀 감정 상태 임베딩 생성
- ERC 모델 구조:
  - RoBERTa 인코더 + MLP 분류기
  - 모든 발화를 단일 시퀀스로 연결, 발화 구분용 [SEP] 토큰 사용:
    $$
    <ucontext> = [SEP]_1, u_1, [SEP]_2, u_2, \ldots
    $$
  - RoBERTa 인코더 후 마지막 은닉층의 각 [SEP] 토큰 임베딩 추출하여 각 발화 표현으로 사용
  - 감정 집합을 $$E$$라고 할 때, 각 발화 $$u_i$$에 대하여:
    $$
    C^{ERC} = \text{RoBERTa}([CLS], <ucontext>)
    $$
    $$
    z_i = \text{MLP}(C^{ERC}[SEP]_i)
    $$
- DailyDialog 데이터셋 기반: 감정 7종 (Ekman 6대 기본 감정 + 중립)
- Mixed-emotion 모듈:
  - 감정 코드북: 학습 가능한 파라미터 행렬 $$E \in \mathbb{R}^{\vert E \vert \times h}$$ ($$h$$: 임베딩 크기)
  - 각 감정 $$E_k$$는 하나의 임베딩 벡터
  - 감정 노드 $$v_i$$의 임베딩 $$g_i^e$$는 감정 분포에 따른 가중합:
    $$
    g_i^e = \mathbf{p}_i \cdot E
    $$
  - 감정 분포 $$\mathbf{p}_i$$ 계산:
    $$
    p_i^j = \frac{\exp(z_i^j / \tau)}{\sum_{k=1}^{|E|} \exp(z_i^k / \tau)}
    $$
  - 학습 가능한 온도 파라미터 $$\tau$$를 통해 분포의 샤프닝 조절

---

### 4.2.2 System Strategy Node Embedding

- 시스템 턴 $$i \in I_{agent}$$ 에 대해 전략 정보는 원-핫 벡터 $$s_i \in \{0,1\}^{|S|}$$
- 전략 임베딩 행렬 $$S \in \mathbb{R}^{|S| \times h}$$
- 시스템 전략 노드 임베딩:
  $$
  g_i^{st} = s_i \cdot S
  $$

---

### 4.2.3 Dummy Node Embedding

- 이전 연구에서는 단순 읽기 함수와 선형 계층으로 이종 그래프 정보 집계함
- 본 모델에서는 더미 노드 $$v_t$$를 예측 대상 자리 표시자로 설정하고, 화자 역할에 맞게 이전 발화들과 상호작용하도록 설계
- 더미 노드 임베딩은 학습 가능한 파라미터 벡터 $$g_d \in \mathbb{R}^h$$ 로 정의되며 모든 그래프에서 공유

---

### 4.2.4 Relational Graph Attention Layers

- 초기 노드 임베딩을 기반으로 관계 그래프 어텐션 네트워크(RGAT) 적용하여 노드 표현 갱신
- 각 엣지 유형 $$r \in R$$ 및 어텐션 헤드 $$k$$에 대해 키, 쿼리, 값 행렬 $$W_K^{(r,k)}, W_Q^{(r,k)}, W_V^{(r,k)}$$ 정의
- 각 노드 쌍 $$(i,j)$$, 엣지 유형에 대한 어텐션 스코어:
  $$
  a_{i,j}^{(r_{ij}, k)} = \sigma\big( W_Q^{(r_{ij}, k)} g_i + W_K^{(r_{ij}, k)} g_j \big)
  $$
  $$
  \alpha_{i,j}^{(r_{ij}, k)} = \frac{\exp(a_{i,j}^{(r_{ij}, k)})}{\sum_{r \in R} \sum_{m \in \mathcal{N}_r(i)} \exp(a_{i,m}^{(r, k)})}
  $$
  - $$\sigma$$: LeakyReLU 함수
  - $$\mathcal{N}_r(i)$$: 노드 $$i$$의 엣지 유형 $$r$$에 해당하는 이웃 노드 인덱스 집합
- 멀티헤드 어텐션 결과:
  $$
  h_i = \mathbin\Vert_{k=1}^K \sigma\left(\sum_{r \in R} \sum_{m \in \mathcal{N}_r(i)} \alpha_{i,m}^{(r,k)} W_V^{(r,k)} g_m\right)
  $$
  - $$\mathbin\Vert$$: 벡터 연결(concatenation)
- 잔차 연결(Residual connection) 적용:
  $$
  g_i^{(1)} = h_i + g_i
  $$
- Dummy 노드의 L번째 RGAT 레이어 임베딩 $$g_N^{(L)}$$를 전체 그래프 표현으로 사용

---

## 4.3 Next Dialogue Strategy Prediction

- 전역 의미 임베딩 $$C[CLS]$$과 이종 그래프의 더미 노드 임베딩 $$g_N^{(L)}$$를 결합(concatenation):
  $$
  o = \text{softmax}\big( \text{MLP}(C[CLS] \mathbin\Vert g_N^{(L)}) \big)
  $$
- 전략 예측 확률 분포 계산:
  $$
  P(s_{t_N} \vert H_1^{N-1}) = o_{s_{t_N}}
  $$
- 학습 손실 함수로 가중치가 조정된 교차 엔트로피 손실(Weighted Cross-Entropy Loss) 사용
  - 클래스 불균형 문제 해결 위해 각 클래스 당 손실 기여도를 클래스 빈도의 역수 비례로 가중치 조정

---

# 5 Experiments

- ## 5.1 Experimental Setups
  - **Datasets**  
    - 두 가지 영어 ESC (Emotional Support Conversation) 데이터셋을 사용하여 사회적 및 과제 지향적 목표에 모두 유익한 전략 학습을 목표로 함.  
      1. **ESConv** (Liu et al., 2021)  
         - 1,300개의 대화, 8가지 대화 전략, 공식 train/dev/test 분할 사용.  
      2. **AnnoMI** (Wu et al., 2022)  
         - 전문가가 주석 작업한 상담 데이터셋, 133개 대화, 9가지 치료사 전략, 8:1:1 비율로 train/dev/test 분할.  
    - 문맥 윈도우 크기는 5문장(발화).  
    - 생성된 샘플 수: ESConv 18,376, AnnoMI 4,442개.  
  - **Baselines**  
    1. **LLM 프롬프트 방식**  
       - ChatGPT4 (OpenAI, 2023), LLaMA3-70B (Meta, 2024)  
       - 2-shot 학습과 감정 레이블(ERC) 정보를 포함한 설정 포함  
       - Chain-of-Thought prompting은 효과가 없음으로 제외  
    2. **LLM 파인튜닝 방식**  
       - RoBERTa, BART, LLaMA3-8B  
    3. **감정 지원 대화 전략 예측을 위한 특화 모델**  
       - MISC, MultiESC, KEMI, TransESC  
  - **평가 지표**  
    - **매크로 F1 점수 (MF1)**, **가중 F1 점수 (W-F1)** 사용 (불균형 데이터셋에 적합).  
    - **선호 편향 점수 (B)**: 모델이 특정 전략에 편향하는 정도를 수치화, 이상적인 예측기는 높은 F1과 낮은 B를 목표.  
  - **구현 세부사항**  
    - PyTorch 기반, RoBERTa 사전학습 가중치 사용, Huggingface 토크나이저 활용.  
    - AdamW 옵티마이저 사용.  
    - 하이퍼파라미터 등 자세한 내용은 부록 D.3 참고.  

- ## 5.2 Overall Performance
  - EmoDynamiX가 모든 평가 지표에서 기존 SOTA 방법들을 뛰어넘음.  
  - TransESC 대비 선호 편향 점수(B)를 38% 감소시키면서 F1 점수는 증가.  
    - TransESC도 대화 상태 전환을 모델링하지만, EmoDynamiX의 혼합 감정(ERC 기반) 접근이 사용자의 감정 상태 변화를 더 세밀하게 포착.  
  - MultiESC 대비 큰 폭으로 모든 지표에서 우수함 (MultiESC는 낮은 편향 점수 보유).  
  - 케이스 지식 기반(KEMI) 및 상식 지식 기반(MISC) 모델은 대화 전략 예측에 덜 효과적임.  
  - LLM 프롬프트 방식은 내재된 편향으로 인해 성능 제약이 크며, 예시나 감정 인식 정보 제공에도 편향 점수가 0.77~1.39로 상당히 높음.  
  - LLM 파인튜닝 중에서는 LLaMA3-8B가 가장 높은 F1 점수 기록, RoBERTa는 편향 점수가 낮은 편, BART는 균형 잡힌 성능 보임.  
  - 그럼에도 불구하고 이들 모두 EmoDynamiX와 비교 시 큰 차이가 있음.  

- ## 5.3 Ablation Study
  - 다음 모듈들의 기여도 검증:  

    1. **대화 맥락에서 사용자 감정 및 에이전트 전략 모델링**  
       - (1) 단순히 flatten된 대화 이력만 사용 (RoBERTa)  
       - (2) 감정 및 전략을 태그 형태로 삽입한 flatten된 컨텍스트 (w/o Graph)  
       - 결과: w/o Graph가 RoBERTa보다 나으나 EmoDynamiX와 큰 차이 존재 → 그래프 학습 모듈이 핵심 역할 수행.  

    2. **혼합 감정 모델링**  
       - 사용자 감정 상태를 원-핫 벡터로만 표현 (w/o Mixed Emotion)  
       - 결과: 모든 지표가 하락 → 감정 분포를 통해 미세한 감정 변화를 포착하는 것이 중요함.  

    3. **담화 구조 모델링**  
       - 노드들을 단순 순차적 연결로 대체 (w/o Discourse Parser)  
       - 결과: 성능 하락 있으나 크지 않음  
       - STAC 담화 파서와 ESC 데이터셋 간 도메인 차이가 효과를 제한하는 원인으로 추정.  

    4. **정보 집계를 위한 더미 노드 사용**  
       - 더미 노드를 평균-최대 풀링으로 대체 (w/o Dummy Node)  
       - 결과: 성능 감소, 특히 AnnoMI에서 매우 크게 나타남 → 저자원 환경에서 더미 노드 설계의 중요성 강조.  

- ## 수식 및 점수 예시
  - 선호 편향 점수는 낮을수록 좋음 (감정 지원 대화 전략 예측 모델의 편향 완화 평가).  
  - 표1에서 EmoDynamiX는 ESConv 데이터셋의 매크로 F1 점수 27.70, 가중 F1 점수 32.71, 선호 편향 점수 0.45를 기록함.  
  - ablation study 표2는 EmoDynamiX 대비 각 구성요소 제거 시 성능 차이를 보여줌 (예: w/o Graph Learning 시 M-F1 1.98 감소, B 0.33 증가).

---

# 6 In-depth Analysis of EmoDynamiX

- 본 연구에서는 ESConv에서 발췌한 사례를 Figure 3을 통해 제시하였음.
- 사례: 
  - 사용자의 감정 상태가 Frustration에서 Joy로 긍정적으로 전환된 후, 에이전트가 어떤 전략을 적용할지 결정하는 상황.
  - Frustration은 DailyDialog에 없는 감정 카테고리이므로, 혼합 감정 모듈에서 약간의 분노가 섞인 중간 정도의 슬픔으로 모델링됨.
  - 실제 정답 전략은 Affirmation and Reassurance로, 사용자의 긍정적 감정 전환을 인정하고 긍정적 기분을 강화하도록 유도함.

- 결정 과정 분석:
  - 더미 노드(dummy node) 엣지의 attention weight를 통해 각 노드가 의사결정에 기여하는 정도를 파악함.
  - RGAT 레이어가 깊어질수록 더미 노드의 관심이 과거에 적용된 전략에서 사용자의 감정 전환 상태로 이동함.
  - 특히 Frustration 노드와 연결된 엣지에 높은 가중치가 부여됨.
  - 이는 그래프 학습 모듈이 감정과 전략 간의 동적 변화를 효과적으로 포착함을 의미함.

- 예측과 인간 전략 간 불일치(disagreement) 분석:
  - 혼동 행렬(confusion matrix)을 사용하여 예측과 실제 전략 간 차이를 조사(Appendix E.3 참고).
  - 감정 관련 대화 전략과 과제(task) 지향 전략의 적절한 적용 시점을 구분하는 것이 어렵다는 점 확인.
  - 모델은 종종 Providing Suggestions (과제 지향 전략)를 예측하지만, 인간은 Refletion of feelings, Self-disclosure, Affirmation and Reassurance (감정 관련 전략)를 선택함.
  - 감정 관련 전략 중 60% 이상이 과제 지향 전략으로 분류되는 현상은 Galland et al. (2022)의 논의와 일치함.

- 감정 노드의 주요 카테고리와 불일치 패턴의 상관관계 분석(Figure 4 참고):
  - 예측과 실제 전략 불일치에 있어 "Neutral" 감정 노드가 상당히 크게 기여함.
  - 이는 ERC 모듈의 전체 출력 분포 내 "Neutral" 비율(59.22%, Appendix E.2 참고)보다 높은 비중임.
  - 결과적으로, 모델이 문맥에서 표현된 사용자의 감정을 잘 포착할수록 전략 예측이 용이함을 시사함.

---

# 7 Conclusions

- 본 논문에서는 EmoDynamiX라는 사회정서적 대화 전략 예측 프레임워크를 제안함.
  - 전문가 모델을 집계하고 이종 그래프(heterogeneous graphs)를 활용하여 사용자 상태와 시스템 전략의 대화 역학을 모델링함.
- 제안 기법은 두 개의 공개된 ESC(감정 사회적 대화 전략) 데이터셋에서 모든 기존 기준선 모델을 크게 능가함.
- 심층 연구에서 주의(attention) 가중치를 분석하여 모델의 투명성에 한걸음 다가섬.

## 한계점

- **진실 데이터(Ground Truth)의 한계**
  - 사용한 ESC 데이터셋은 인간 평가를 거쳤으나, 동일 문맥에서 진실 데이터 이외의 다른 전략이 효과적일 가능성을 완전히 배제할 수 없음.
  - 전략 계획 단계에서의 인간 평가 프로토콜이 부족하며, 인간 평가는 생성 단계 이후에 수행하는 것이 더 적합함.

- **다른 언어에 대한 일반화 가능성**
  - 제안한 구조는 영어 데이터셋 두 개로만 평가되었음.
  - 다른 언어나 다중 언어 환경에 대한 일반화 가능성은 아직 미확인 상태임.
  - EmoDynamiX는 영어 데이터셋 기반의 전문가 모델에 의존하므로, 해당 데이터셋에 내재된 문화적 편향이 전략 예측에 영향을 미칠 수 있음 (예: Gelfand et al., 2011; Hall, 1976).

- **전문가 모듈의 한계**
  - 감정 인식(ERC)과 담화 분석은 본 연구의 주된 기여가 아니어서, 다양한 모델 설계나 데이터셋이 결과에 미치는 영향에 대해서는 다루지 않음.
  - 향후 연구에서 도메인 간 ERC 모듈과 담화 파서의 학습 및 통합을 고려할 수 있음.

- **실용 적용과의 거리**
  - 제안 방법이 기존 기준선보다 성능이 뛰어나지만, 여전히 만족스럽지 않은 수준임.
  - 이는 해당 과제의 복잡성을 반영하며, 향후 ESC 에이전트에서 사회정서 전략 예측기를 견고한 구성 요소로 만들기 위한 추가 연구가 필요함.

## 윤리적 고찰

- **기술의 의도**
  - 대화형 AI는 인간을 대체하기 위해 개발되어서는 안 되며, 인간과 AI의 명확한 경계를 유지하는 것이 중요함 (Ethique et al., 2024).
  - 학습 데이터가 인간에 의해 선별된 만큼, AI는 인간과 유사한 행동을 할 수 있음.
  - 예를 들어, 그림 1에서 시스템은 ‘자기노출(Self-disclosure)’ 전략으로 외로움을 표현하는데, 이는 윤리적 권고와는 상충됨.
  - 이러한 인간 같은 행동을 유발하는 전략은 실제 환경에서 주의 깊게 적용하거나 제한할 필요가 있음.
  - 본 연구의 접근법은 AI의 바람직한 행동을 명시적으로 설정할 수 있어, 향후 대화형 AI의 제어력을 높이는 데 기여할 수 있음.

- **데이터 프라이버시**
  - 모든 실험은 공개된 과학적 연구 데이터셋으로 수행되었으며, 개인정보 및 민감한 정보(사용자 및 플랫폼 식별자 등)는 제거됨.

- **의료 관련 면책**
  - 본 연구는 치료 권고나 진단 주장 등을 제공하지 않음.

- **투명성**
  - 데이터셋 통계 및 하이퍼파라미터 설정을 상세히 기술하였으며, 분석 결과는 실험 결과와 일치함.

## 감사의 말

- 익명의 리뷰어들의 귀중한 의견에 감사함.
- 본 연구는 ANR-23CE23-0033-01 SINNet 프로젝트와 프랑스 2030 계획 하의 ANR-23-IACL-0008 프로젝트의 일부 지원을 받음.

---

# References

- Alafate Abulimiti, Chloé Clavel, Justine Cassell (2023). end-to-end 신경망 모델을 이용한 발화 완곡 표현 생성 연구. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)*.

- Nicholas Asher et al. (2016). 다자간 대화에서 담화 구조와 대화 행위 연구: STAC 코퍼스. *LREC 2016*.

- Anna Braniecka et al. (2014). 복합 감정과 대처 전략: 2차 감정의 이점. *PloS One*, 9(8):e103940.

- Dan Busbridge et al. (2019). 관계 그래프 어텐션 네트워크. *arXiv preprint arXiv:1904.05811*.

- Carlos Busso et al. (2008). IEMOCAP: 상호작용 감정 이중 모션 캡처 데이터베이스. *Language Resources and Evaluation*, 42:335–359.

- Feiyu Chen et al. (2023a). 다변량·다중 주파수·다중 모달 정보 기반 대화 감정 인식을 위한 그래프 신경망 재고찰. *CVPR 2023*.

- Jiaao Chen, Diyi Yang (2021). 담화 및 행동 그래프 기반 구조 인식 추상적 대화 요약. *NAACL-HLT 2021*.

- Maximillian Chen et al. (2023b). 프롬프트를 통한 제어 가능한 혼합 주도 대화 생성. *ACL 2023, Volume 2*.

- Yi Cheng et al. (2022). 미래 예측 전략 계획을 통한 다중턴 감정 지원 대화 생성 개선. *EMNLP 2022*.

- Cyril Chhun, Fabian M. Suchanek, Chloé Clavel (2024). LLM에 대한 자동 스토리 평가를 위한 프롬프트. *Transactions of the Association for Computational Linguistics*.

- Ta-Chung Chi, Alexander Rudnicky (2022). 구조화된 담화 분석. *SIGDIAL 2022*.

- Chloé Clavel, Matthieu Labeau, Justine Cassell (2022). 사회대화시스템의 세 가지 도전과제. *Frontiers in Robotics and AI*.

- Sumanth Dathathri et al. (2020). 제어된 텍스트 생성을 위한 플러그 앤 플레이 언어 모델. *ICLR 2020*.

- Yang Deng et al. (2024). 대규모 언어 모델 기반 대화 에이전트를 위한 플러그 앤 플레이 정책 플래너. *ICLR 2024*.

- Yang Deng et al. (2023). 지식 기반 혼합 주도 감정 지원 대화 시스템. *ACL 2023*.

- Comité Ethique et al. (2024). 사회 로봇 애착 현상과 과학 공동체에 대한 경계 촉구. CNRS COMETS 기술 보고서.

- Xiachong Feng et al. (2021). 회의 요약을 위한 담화 인식 그래프 모델 및 데이터 증강. *IJCAI 2021*.

- Changzeng Fu et al. (2023). 계층적 어텐션과 그래프 네트워크를 통한 대화 행위 분류. *ICASSP 2023*.

- Lucie Galland et al. (2022). 정보 제공 인간-에이전트 상호작용에서 대화 전략 적응. *Frontiers in AI*.

- Michele J Gelfand et al. (2011). 엄격 문화와 느슨한 문화의 차이: 33개국 연구. *Science*, 332(6033):1100–1104.

- Deepanway Ghosal et al. (2019). 대화 감정 인식을 위한 DialogueGCN. *EMNLP-IJCNLP 2019*.

- Edward T Hall (1976). 문화 너머. *Anchor*.

- Geoffrey Hinton et al. (2015). 심층 신경망 지식 증류. *NIPS Workshop*.

- Edward J Hu et al. (2021a). LORA: 대형 언어 모델 저랭크 적응. *arXiv:2106.09685*.

- Jingwen Hu et al. (2021b). MMGCN: 감정 인식을 위한 다중 모달 융합 딥 그래프 합성곱 네트워크. *ACL/IJCNLP 2021*.

- Jena D Hwang et al. (2021). (comet-) atomic 2020: 상징적 및 신경 상식 지식 그래프. *AAAI 2021*.

- Taichi Ishiwatari et al. (2020). 관계 인지 그래프 어텐션 네트워크 감정 인식. *EMNLP 2020*.

- Rishabh Joshi et al. (2021). 협상 대화에 전략-그래프 네트워크 통합: Dialograph. *ICLR 2021*.

- Dongjin Kang et al. (2024). 대형 언어 모델이 좋은 감정 지원자가 될 수 있는가? 선호 편향 완화. *ACL 2024*.

- Mike Lewis et al. (2020). BART: 잡음 제거 seq-to-seq 사전학습. *ACL 2020*.

- Ge Li et al. (2024). DQ-HGAN: 이종 그래프 어텐션 및 심층 Q-러닝 기반 감정 지원 대화 생성. *Knowledge-Based Systems*.

- Wei Li et al. (2023). SKIER: 기호 지식 통합 모델 대화 감정 인식. *AAAI 2023*.

- Yanran Li et al. (2017). DailyDialog: 수작업 라벨링된 다중턴 대화 데이터셋. *IJCNLP 2017*.

- Siyang Liu et al. (2021). 감정 지원 대화 시스템 방향. *ACL/IJCNLP 2021*.

- Xin Liu et al. (2022). 더미 노드 활용 그래프 구조 학습 강화. *ICML 2022*.

- Yinhan Liu et al. (2019). RoBERTa: 견고하게 최적화된 BERT. *arXiv:1907.11692*.

- Ilya Loshchilov, Frank Hutter (2019). 분리된 가중치 감쇠 정규화. *ICLR 2019*.

- Yiming Lu et al. (2024). Strux: 구조화된 설명 기반 의사결정 LLM. *arXiv:2410.12583*.

- Josh Magnus Ludan et al. (2023). 설명 기반 미세조정으로 모델 내 잡음 신호 내성 향상. *ACL 2023*.

- Meta (2024). Meta LLaMA 3. https://llama.meta.com/llama3/.

- OpenAI (2023). ChatGPT. https://openai.com/blog/chatgpt.

- Adam Paszke et al. (2019). PyTorch: 고성능 딥러닝 라이브러리. *NeurIPS 2019*.

- Wei Peng et al. (2022). 전역 제어·지역 이해: 전역-지역 계층형 그래프 네트워크 E-support 대화. *IJCAI 2022*.

- Soujanya Poria et al. (2019). MELD: 다중 모달 다자간 감정 인식 데이터셋. *ACL 2019*.

- Aritra Raut et al. (2023). 감정 강화 그래프 주의력 모듈 기반 협상 대화 생성. *IJCNLP/APACL 2023*.

- Nils Reimers, Iryna Gurevych (2019). Sentence-BERT: Siamese BERT 기반 문장 임베딩. *EMNLP-IJCNLP 2019*.

- Franco Scarselli et al. (2008). 그래프 신경망 모델. *IEEE Transactions on Neural Networks*, 20(1):61–80.

- Weizhou Shen et al. (2021). 대화 감정 인식을 위한 방향성 비순환 그래프 네트워크. *ACL/IJCNLP 2021*.

- Quan Tu et al. (2022). MISC: COMET 통합 혼합 전략-aware 모델 ESC. *ACL 2022*.

- Janneke M van der Zwaan et al. (2012). 지능형 에이전트 위한 감정 지원 대화 모델. *Modern Advances in Intelligent Systems and Tools*.

- JM Van der Zwaan et al. (2012). BDI 대화 에이전트 설계 및 평가. *AAMAS Workshop 2012*.

- Lorraine Vanel et al. (2023). 과제 지향 대화에서 감정 및 대화 전략 예측 과제. *ACII 2023*.

- Dong Wang et al. (2020). 사용자 이력 통합 이종 그래프로 대화 행위 인식. *COLING 2020*.

- Jason Wei et al. (2024). 사슬 사고 프롬프팅을 통한 대형 언어 모델 추론 능력 유도. *NeurIPS 2022*.

- Thomas Wolf et al. (2020). Transformers 라이브러리, 최첨단 자연어처리. *EMNLP 2020 Demonstrations*.

- Zixiu Wu et al. (2022). Anno-MI: 전문가 주석 상담 대화 데이터셋. *ICASSP 2022*.

- Kailai Yang et al. (2023). 상식 지식을 활용한 감정 추론 향상을 위한 이분 그래프. *CIKM 2023*.

- Sayyed M Zahiri, Jinho D Choi (2018). TV 쇼 대본 기반 감정 탐지 CNN. *AAAI 워크숍*.

- Duzhen Zhang et al. (2023). DualGATs: 대화 감정 인식 위한 이중 그래프 어텐션 네트워크. *ACL 2023*.

- Weixiang Zhao et al. (2023). TransESC: 턴 단위 상태 전이로 ESC 향상. *ACL 2023 Findings*.

- Zhonghua Zheng et al. (2023). 대형 언어 모델 시대의 감정 지원 챗봇 구축. *arXiv:2308.11584*.