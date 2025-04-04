---
title: "[논문리뷰] Aspect category sentiment analysis based on prompt-based learning with attention mechanism (Neurocomputing)"
date: 2025-04-04 00:00:00 +0900
categories:
  - Paper Review
tags:
  - ABSA
  - Neurocomputing
---

이 논문은 특정 측면의 감정 극성을 평가하는 세 가지 주요 구성 요소(측면 용어 추출, 측면 범주 탐지, 감정 분류)를 활용하여 새로운 프롬프트 기반 공동 모델(PBJM)을 제안합니다. 이 모델은 측면 범주 분석(ACSA) 작업을 개선하여 감정 분석의 정확성을 높이고, 기존 모델보다 우수한 성능을 나타냄을 보여줍니다.

---

# 1. Introduction

<img width="808" alt="image" src="https://github.com/user-attachments/assets/c6989997-4a25-453f-a8d0-90b4436ff526" />

- **ABSA(Aspect-Based Sentiment Analysis)**는 문장에서 특정 측면(aspect)에 대한 감정 극성(sentiment polarity)을 분석하는 세분화된 감성 분석 과제이다.

- ABSA는 세 가지 하위 과제로 구성됨:
  - **ATE (Aspect Term Extraction)**: 문장에 명시된 측면 단어를 추출
  - **ACD (Aspect Category Detection)**: 문장에서 언급된 측면 카테고리 감지
  - **SC (Sentiment Classification)**: 특정 측면(또는 카테고리)에 대한 감정 극성 예측

- **ACSA(Aspect Category Sentiment Analysis)**는 ACD와 SC를 결합하여 명시적 단어 없이도 카테고리 감정 예측 가능 (ex. “Bagels are ok, but be sure not to make any special requests!”)

- **기존 방법의 문제점**:
  - Cartesian Product 방식: 모든 카테고리-감정 쌍을 생성해 분류 ⇒ **데이터 과도 증가**
  - Binary 확장 방식: 쌍 존재 여부를 감정 분류에 포함 ⇒ **복잡도 증가 및 불필요한 정보 발생**
  - 계층적 분류(Hierarchical classification): 카테고리 감지 후 감정 예측하지만 **두 작업 간 관계를 반영하지 못함**

- **본 논문의 핵심 제안**:
  - **PBJM (Prompt-Based Joint Model)**: 프롬프트 학습과 PLM 기반 공동 학습으로 문제 해결
    - 프롬프트를 활용해 **카테고리 감지를 간결하게 수행**
    - 감정 분류는 **이진 분류**로 처리해 단순화
    - 두 하위 과제를 **BERT 기반으로 공동 학습**, 상호 연관성 강화
    - **어텐션 메커니즘**으로 중요한 단어에 집중
    - **버벌라이저(verbalizer)**를 통해 라벨 단어와 라벨 공간 연결

- **기여 및 성과**:
  - 프롬프트 학습 기반 PBJM 모델을 처음으로 제안
  - 감정 분류 다중 클래스 문제를 멀티 레이블 문제로 전환
  - **4개 벤치마크 데이터셋에서 기존 모델들보다 높은 성능** 달성

---

# 2. Related works

- ACSA 작업을 해결하는 가장 직관적인 방법은 파이프라인 접근 방식:
  - 문장에서 언급된 카테고리 감지 및 해당 카테고리에 대한 감정 극성 예측 포함.
  
- 카테고리 감지 방법:
  - Movahedi et al. [2]: 문장의 특정 요소에 주의를 기울이는 방법 사용.
  - Ghadery et al. [3]: 각 카테고리의 대표 단어와 문장을 비교하여 카테고리 존재 여부 결정.

- 감정 극성 예측 방법:
  - Wang et al. [4]: 문맥과 카테고리 간 관계를 모델링하는 방법 제안, 주의 향상 LSTM 사용.
  
- 파이프라인 접근 방식의 한계:
  - 초기 단계의 오류가 전체 예측 정확도에 상당한 영향을 미칠 수 있음.
  - 카테고리 탐지와 감정 예측 간의 연결을 무시하는 경향 존재, 이는 중요한 문제.

- 통합된 프레임워크 접근:
  - Cartesian Product 접근 [7]: 카테고리-감정 쌍의 모든 가능 조합 생성.
  - Schmitt et al. [8]: 문장에서 카테고리 존재 여부를 나타내기 위해 "N/A" 차원 도입.
  
- Hierarchical Graph Convolutional Network (Hier-GCN):
  - Cai et al. [10]: 두 개의 GCN 레이어 사용, 첫 번째 레이어는 상관관계 탐지, 두 번째 레이어는 감정 극성 예측.
  
- SRGN(Semantic Relatedness-Enhanced Graph Network):
  - Zhou et al. [12]: Edge-Gated Graph Convolutional Network (EGCN) 사용, 의미 관련 정보 통합.
  
- 공유 매개변수 감정 예측 레이어:
  - Li et al. [13]: 서로 다른 카테고리 간 감정 지식 전파 및 특정 카테고리의 데이터 부족 완화.
  
- 이중 주의 메커니즘:
  - Gu와 Zhang [14]: 분석에서 카테고리와 감정을 효과적으로 고려.

- 현재 연구의 한계:
  - ACD(Aspect Category Detection)와 SC(Sentiment Classification) 서브작업 간의 관계를 잘 모델링하지 못함.
  
- 제안:
  - 이 문제를 해결하기 위한 공동 학습 접근 방식 제안.

---

# 2.1. Aspect category sentiment analysis

- **프롬프트 기반 학습 연구**  
  - PLMs(사전학습된 언어 모델)를 활용하여 사전 학습 중 습득한 지식을 활용하려는 연구가 진행됨.
  
- **자연어 프롬프트 사용**  
  - Seoh et al.의 연구에서는 제로샷 및 퓨샷 사례에 자연어 프롬프트를 활용.
  - LAMA 데이터셋은 인간이 제작한 클로즈 템플릿을 사용하여 PLMs가 사전 학습 중 습득한 지식을 회수하도록 자극.
  
- **프리트레인된 프롬프트 템플릿 활용**  
  - Schick과 Schütze는 사전 훈련된 프롬프트 템플릿을 활용하여 텍스트 분류 및 조건에 따른 텍스트 생성을 수행.
  
- **자동화된 템플릿 디자인 방법**  
  - Jiang et al.는 마이닝 기반 및 패러프레이징 기반 방법을 통해 자동 프롬프트 템플릿 생성 제안.
  - Haviv et al.는 BERT 기반 리라이터를 사용하여 PLMs의 지식 추출 수행.
  - Gao et al.는 T5를 활용하여 프롬프트 생성을 텍스트 생성 과제로 취급.

- **연속 템플릿 사용 탐색**  
  - Qin과 Eisner는 경량 템플릿 혼합을 조정하기 위해 경량 하강법을 활용.
  - "P-tuning"은 Liu et al.에 의해 제안되어, 연속 자유 매개변수를 삽입하여 자동으로 프롬프트를 검색.

- **지식 주입을 통한 성능 향상**  
  - Li et al.는 감정 지식을 사용하여 통합 네트워크 내에서 프롬프트를 향상.
  - Hu et al.는 외부 지식을 언어화 도구에 통합하여 프롬프트 기반 학습 성능 개선.

---

# 3. Methods

<img width="808" alt="image" src="https://github.com/user-attachments/assets/89a4488c-5441-46f4-b40b-f48632634764" />

- 리뷰 문장이 n개의 단어로 구성되어 있음 (𝑊 = [𝑤1, 𝑤2, . . . ,𝑤𝑛])
- 미리 정의된 측면 카테고리 집합 𝐶와 감정 극성을 포함하는 집합 𝑃 (긍정, 부정, 중립) 존재
- ACSA(Aspect Category Sentiment Analysis)의 목표:
  - 각 입력 𝑊에 대해 카테고리와 감정의 모든 가능한 조합 생성 ($$c, p$$), 
  - 여기서 $$c$$는 집합 𝐶의 측면 카테고리 레이블, $$p$$는 감정 극성
- ACSA 문제를 해결하기 위한 PBJM 모델 구성 요소:
  - BERT 인코더 모듈
  - 주의 메커니즘 모듈
  - 음성화 모듈
  - 감정 분류기 모듈
- 이 섹션의 나머지 부분에서는 각 구성 요소에 대한 자세한 정보를 제공함.

---

# 3.1. Task definition

- 해당 모듈에서는 문장에서 측면 범주를 식별하기 위해 `"the sentence is about the [𝑀𝐴𝑆𝐾]"`라는 프롬프트 템플릿을 수동으로 생성함.
- 입력 시퀀스는 다음과 같이 형성됨:  
  $$ S = [C_{LS}] \, sentiment\_polarity [S_{EP}] \, original \, sentence + template [S_{EP}] $$
- 여기서 [𝐶𝐿𝑆]와 [𝑆𝐸𝑃]는 특수 토큰이며, ‘sentiment’는 다양한 감정 극성을 가리킴.
- Eq. (1)에 따르면, 원본 데이터셋의 각 문장은 동일한 프롬프트 템플릿을 가지지만 서로 다른 감정 극성을 가진 세 개의 문장으로 확장됨.
- 훈련 과정은 확장된 새 데이터셋에서 수행됨.
- 입력 시퀀스는 BERT 모델을 이용해 숨겨진 상태로 변환됨:  
  $$ T = BERTEncoder(S) $$
- 여기서 $$ T \in R^{(n+10) \times d_{m}} $$이고, $$ d_{m} $$은 BERT 모델의 숨겨진 차원임.

---

# 3.2. BERT encoder module

- **목적**: 카테고리 주의(category attention)는 문장의 중요한 구간을 식별하여 특정 카테고리에 맞는 내용을 추출.
  
- **벡터 표현**:
  - 원본 문장: $$T_W \in \mathbb{R}^{n \times d_m}$$
  - 토큰 [MASK]: $$T_M \in \mathbb{R}^{d_m}$$

- **카테고리 주의 정의**:
  - $$H = \sum_{i=1}^{n} T_W^i \cdot T_M \left( \sum_{j=1}^{n} T_W^j \cdot T_M \cdot T_W^i \right)$$

- **후처리**:
  - $$H' = \text{dense}(T_M \oplus H)$$
  - 여기서 $$H \in \mathbb{R}^{d_m}$$, $$H' \in \mathbb{R}^{d_m}$$, $$n$$은 문장 단어 수.
  
- **벡터 연결**:
  - $$\oplus$$는 두 벡터를 연결하는 연산.

- **풀리 연결 레이어 정의**:
  - $$\text{dense}(x) = \tanh(W_d x + b_d)$$
  - $$W_d \in \mathbb{R}^{d_m \times 2d_m}$$, $$b_d \in \mathbb{R}^{d_m}$$는 가중치와 편향을 나타냄.

---

# 3.3. Category attention module

- **목적**: 
  - 프롬프트 학습의 언어화기는 모델에서 가져온 레이블 단어를 필요한 레이블로 매핑.
  - 레이블 단어의 품질이 최종 예측 결과에 직접적인 영향.

- **동의어 사용**:
  - 동의어를 사용하여 레이블의 범위를 확장하고 원래 의미 유지.
  - thorough하고 우수한 품질의 레이블 단어 확보.

- **레이블 구조**:
  - “레이블”은 특정 aspect 카테고리와 연결.
  - 구성 요소: 엔터티 (E)와 속성 (A)로 나뉘어짐.
  - 예: ‘‘service#general’’에서, ‘‘service’’는 E, ‘‘general’’은 A.

- **벡터 표현**:
  - 벡터 표현 𝐻′를 BERT 사전 훈련된 선형 층을 통과시켜 확률로 변환.
  - 표현식: $$H_v = \tanh(W_v \cdot H' + b_v)$$ 
    - 여기서 $$H_v \in \mathbb{R}^{d_v}, W_v \in \mathbb{R}^{d_v \times d_m}, b_v \in \mathbb{R}^{d_v}$$ 
    - $$d_v$$는 어휘 단어 수.

- **확률 추출**:
  - 모든 확률 대신 $$i$$번째 레이블 단어의 확률만 추출 가능.
  - 표현식: $$H_l^i = \text{extr}(H_v)$$
    - 여기서 $$H_l^i \in \mathbb{R}^{d_l}$$ 
    - $$d_l$$는 레이블 단어의 길이.

- **점수 할당**:
  - $$i$$번째 aspect 카테고리에 점수 할당.
  - 표현식: $$p_c^i = \sigma(\text{softmax}(W_C^i \cdot H_l^i))$$ 
    - 여기서 $$p_c^i \in \mathbb{R}^{2}$$, $$W_C^i \in \mathbb{R}^{2 \times d_l}$$ 
    - $$N$$은 aspect 카테고리 수, $$\sigma(\cdot)$$는 점수의 연결 작업.

- **실험 데이터셋 통계**:
  - 각 데이터셋의 카테고리, 문장 수, 긍정/중립/부정 수 포함.

---

# 3.4. Category verbalizer module

<img width="455" alt="image" src="https://github.com/user-attachments/assets/fb3a6c43-bb5a-41e3-af2e-c52011c043b1" />

- **목적**: 첫 번째 토큰 [CLS]의 벡터 $$T_{CLS} \in \mathbb{R}^{d_{m}}$$을 이용하여 감정 극성을 위한 이진 레이블 예측
- **구성 요소**:
  - Fully Connected (FC) 레이어
  - 소프트맥스 디코더
- **수식**:
  - 예측 확률 $$p_{s} = \text{softmax}(W_{s} \cdot T_{CLS} + b_{s})$$ (식 9)
- **변수 설명**:
  - $$p_{s} \in \mathbb{R}^{2}$$: 예측된 확률
  - $$W_{s} \in \mathbb{R}^{2 \times d_{m}}$$: 학습 가능한 파라미터
  - $$b_{s} \in \mathbb{R}^{2}$$: 학습 가능한 파라미터

---

# 3.5. Sentiment classifier module

- **정의**:
  - 동질적인 감정 극성을 가진 𝑖번째 감정-카테고리 쌍의 예측: 
    $$(\hat{y}_s, \hat{y}_{c_i}) = (\text{arg max} p_s, \text{arg max} p_{c_i})$$

- **예측 과정**:
  - 다양한 감정 극성 하의 카테고리 분포 예측을 위해 세 번 반복.

- **손실 함수 구성**:
  1. **감정 분류기 모듈**:
     - 사용되는 손실 함수: 이진 교차 엔트로피 
     - 정의: 
       $$\text{loss}_s = -2 \sum_{i=1} y_{s_i} \log p_{s_i}$$
       - 여기서 $y_{s_i}$는 진실 라벨.
  
  2. **카테고리 언어 모델 모듈**:
     - 멀티 레이블 분류를 위한 손실 함수: 
     - 정의: 
       $$\text{loss}_c = -N \sum_{i=1}^N \sum_{j=1}^2 y_{c_{ij}} \log p_{c_{ij}}$$
       - 여기서 $y_{c_{ij}}$는 진실 라벨, $N$은 카테고리 수.
  
- **총 손실 함수**:
  - 총 손실은 두 부분의 합으로 구성됨: 
    $$Loss = \text{loss}_s + \text{loss}_c + \lambda ||\theta||^2$$
    - 여기서 $\lambda$는 $L^2$ 정규화 계수.

---

# 4. Experiments

<img width="749" alt="image" src="https://github.com/user-attachments/assets/be18d2a3-8153-4f43-98dc-23499b055efb" />

- 실험에 사용된 데이터셋:
  - SemEval-2015 및 SemEval-2016에서 수집된 4개의 공개 데이터셋 
  - 식당 및 노트북 도메인의 리뷰 포함
  - 데이터셋 통계는 표 2에 표시
    - 미리 정의된 측면 카테고리 수
    - 문장 수
    - 긍정, 중립, 부정 문장 수

- 실험 설정:
  - 인코더로서 768차원의 비케이스 BERT 사용
  - 최대 문장 길이 108로 설정
  - 에폭 수: 30, 배치 크기: 8
  - verbalizer 모듈의 매개변수 $$W_C$$ 학습률: $$2 \times 10^{-3}$$
  - 다른 파라미터 학습률: $$2 \times 10^{-5}$$
  - 드롭아웃 비율: 0.1, $$L_2$$ 정규화: $$1 \times 10^{-4}$$
  - 평균 점수는 5회 실행하여 안정적인 결과 제공

- 실험 결과:
  - ACSA 작업의 결과 (표 3)
    - 다양한 모델(PR, R, F1 점수)
    - 최고 성능 모델: Our PBJM

- Ablation 연구 결과 (표 4):
  - PBJM 모델의 변형(Attention, prompt, joint 제거에 따른 성능 비교)
  - 각 변형과 PBJM 모델의 성능 차이 확인

- 최종 평가 지표:
  - 예측 결과의 정밀도(Precision)와 재현율(Recall) 계산
  - 마이크로 F1 점수 사용

---

# 4.1. Datasets and experimental settings

- ACSA 작업에서 여러 접근 방식 간 비교 분석 수행
- 사용된 방법들:
  - **Pipeline-BERT**: ACD 및 SC를 위한 파이프라인의 인코더로 BERT 사용
  - **Cartesian-BERT**: 문장 인코더로 BERT를 활용하고, 카르테시안 곱 방식 적용
  - **Addonedim-BERT**: BERT를 문장 인코더로 사용하며 add-one-dimension 방법 활용
  - **AS-DATJM**: ACD에서 주의 메커니즘 활용하여 측면 범주에 대한 벡터 표현 얻음. 이 벡터를 기반으로 GCN을 통해 SC 작업의 감정 맥락 집계
  - **Hier-BERT**: BERT를 인코더로 사용하는 계층적 예측 방법
  - **Hier-Trans-BERT**: Hier-BERT를 기초로 하여 Transformer를 사용해 측면 범주 간의 연결 및 해당 범주의 감정 간 상관관계 포착
  - **Hier-GCN-BERT**: Hier-BERT를 기반으로 관계 학습을 위해 GCN 사용
  - **PBJM**: 수동 프롬프트 템플릿과 주의 메커니즘을 결합한 공동 모델 제안

---

# 4.2. Compared methods

- **모델 성능**:
  - 각 모델은 다양한 데이터셋에서 개별적인 장단점을 보임.
  
- **Pipeline-BERT**:
  - 높은 재현율(recall)을 보이나, 정확도(precision)는 낮음.
  - 두 개의 하위 작업 간 제한으로 인해 발생함.
  
- **Cartesian-BERT**:
  - 레스토랑 도메인에서 성능 우수.
  - 노트북 도메인에서는 카테고리가 많아 어려움 겪음.
  
- **Addonedim-BERT 및 AS-DATJM**:
  - Cartesian-BERT와 유사한 문제를 직면함.
  
- **공동 모델(Hier-BERT, Hier-Tans-BERT, Hier-GCN-BERT)**:
  - 양면성(category)과 감정(sentiment) 간의 상호작용을 고려하는 중요성을 강조.
  - 하지만 두 결과 간의 불일치 가능성도 존재.

- **PBJM 모델**:
  - BERT 인코더를 활용하여 측면 카테고리 탐지(aspect category detection)와 감정 분류(sentiment classification) 간의 관계를 공동으로 설정.
  - 프롬프트 기반 학습(prompt-based learning) 사용하여 PLM의 효율성 극대화.
  
- **데이터셋 규모 감소**:
  - 측면 카테고리 탐지를 위해 특별히 설계된 템플릿 사용.
  
- **F1 성능**:
  - PBJM 모델, 추가 방법 없이도 다른 기준 시스템에 비해 F1 성능에서 우수함.
  
- **미래 연구 및 실제 응용 가능성**:
  - 프롬프트 기반 패러다임이 PLM 성능을 향상시킬 수 있는 잠재력 존재.

---

# 4.3. Main results

- **모델 PBJM에 대한 Ablation 실험 수행**
  - 각각의 모델 구성 요소의 유효성을 확인하기 위함.
  
- **PBJM w/o Attention**
  - 주의(attention) 메커니즘 제거.
  
- **PBJM w/o prompt**
  - 프롬프트 기반 학습 제거.
  - 템플릿을 토큰 [MASK]로 교체하고, Verbalizer를 3층 완전 연결 네트워크로 대체.

- **PBJM w/o joint**
  - ACD와 SC를 각각 별도로 훈련.
  - ACD는 프롬프트 기반 학습을 활용.

- **결과 요약**
  - 다양한 기법이 모델에 유익함을 보여줌.
    - **주의 메커니즘의 효과**: 모델이 문장 내 숨겨진 측면 용어를 식별.
    - **프롬프트 기반 학습의 효과**: 특히 노트북 도메인에서 모델 성능 크게 향상, 어려운 측면 범주를 구별하는 데 도움.
    - **Joint Training의 효과**: 두 하위 작업 간의 관계를 캡처, 모델 성능을 향상시키는 효과적 다리 역할.

- **결론**
  - 주의 메커니즘, 프롬프트 기반 학습, 및 Joint Training의 결합이 ABSA를 위한 더 강력하고 효과적인 모델을 구축.
  - 이러한 기법을 활용하여, 모델은 측면 범주와 감정 간의 연결을 보다 잘 이해하고 활용하게 되어, 다양한 도메인 및 데이터셋에서 성능 향상.

---

# 4.4. Ablation study

- 다양한 템플릿이 ACSA 작업에 미치는 영향을 조사함.
- 다음과 같은 프롬프트 템플릿을 제시:
  - 템플릿 (0): 실험에 사용된 템플릿
  - 템플릿 (1): "문장은 [MASK]라고 생각함"
  - 템플릿 (2): "[MASK]는 문장의 카테고리"
  - 템플릿 (3): "문장의 [MASK]에 관한 것"

- **표 5**: 각 템플릿이 두 개의 훈련 데이터셋에 미치는 영향
  - Rest15에서 템플릿 (0) 사용 시 데이터셋 크기가 3배 증가
  - 템플릿 (1)는 각 카테고리 수에 따라 데이터셋을 증가시킴, 더 많은 카테고리가 있을수록 중복 정보가 증가함

- **표 6**: 다양한 템플릿 결과
  - 템플릿 (1): F1 점수 감소, 특히 Lap15 데이터셋에서 두드러짐
  - 템플릿 (0) 선택 이유: 중복 정보가 결과에 부정적 영향 미침
  - 템플릿 (2)와 (3)은 마스킹 위치는 다르지만 비슷한 최종 결과 도출
  - 훈련 반복 횟수 증가에 따라 템플릿의 최종 결과에 대한 영향이 점차 줄어듦

- **그림 3**: Rest15 데이터셋에서 다양한 템플릿 (0), (2), (3)의 성능 변화
  - 초기 훈련 시 다양한 템플릿 간 결과 차이 있음
  - 훈련 에포크 수가 증가함에 따라 결과 차이가 점차 줄어듦
  - 다양한 템플릿에 의한 영향도 시간이 지남에 따라 감소함

---

# 4.5. Impact of templates

- 실험을 통해 모델의 강건성을 입증하기 위해 여러 전형적인 케이스가 선택됨
  - (a) 명시적 측면
  - (b) 암시적 측면
  - (c) 동일한 감정을 가진 다중 카테고리
  - (d) 다중 감정 카테고리를 가진 다중 카테고리

- 테스트 결과(Table 7):
  - 모델이 명시적 및 암시적 측면 카테고리를 정확하게 식별
  - 여러 카테고리가 동일한 감정을 가질 때도 정확한 예측 가능
  - 다양한 시나리오에서 감정 분석 모델의 강건성과 신뢰성 입증

- 예시 (e)에서 4개의 카테고리 중 2개 식별에 어려움:
  - 첫째, 복잡한 의미적 관계로 인해 암시적 측면 카테고리 확인의 어려움
  - 둘째, 카테고리의 라벨 단어 정의가 어려워 혼란을 초래할 가능성
    - 이는 사용된 언어의 복잡성, 카테고리의 모호성, 훈련 데이터 부족 등 여러 요인에 기인할 수 있음.

---

# 4.6. Case study and error analysis

- 제안된 연구는 PBJM 모델을 소개함.
- PBJM 모델의 목표:
  - ACSA 작업의 한계를 극복하기 위해 프롬프트 기반 학습 사용.
- 모델 특징:
  - 사전 학습에서 멀티 클래스 분류를 멀티 레이블 분류로 전환.
  - 카테고리 주의를 활용하여 관련 카테고리 정보를 획득.
- 실험 결과:
  - 네 개의 벤치마크 데이터셋에서 기존 모델들을 크게 초과 성능을 나타냄.
- 연구 결과:
  - PLM(미리 학습된 언어 모델) 내 정보의 중요성 강조.
  - 작업 유형보다 정보의 양이 더 중요하다는 점 부각.

---

# 5. Conclusion

- **연구 목적**:  
  - ACSA(Aspect Category Sentiment Analysis)의 기존 한계를 프롬프트 기반 학습으로 해결

- **모델 특징**:  
  - 기존 **사전학습 모델의 다중 클래스 분류**를  
    → **다중 라벨 분류(multi-label classification)**로 전환  
  - **Category Attention**을 활용하여  
    → 문장에서 **측면 카테고리 정보**를 효과적으로 추출

- **실험 결과**:  
  - 네 개의 **벤치마크 데이터셋**에서 실험 수행  
  - 기존의 최신 모델들을 **유의미하게 능가하는 성능** 달성

- **핵심 시사점**:  
  - **사전학습 언어 모델(PLM)**에 내재된 정보가  
    → 과제(Task) 자체보다 **더 중요한 요소**로 작용함을 입증

---
