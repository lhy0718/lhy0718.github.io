---
title: "[논문리뷰] Investigating the Personality Consistency in Quantized Role-Playing Dialogue Agents (EMNLP 2024)"
date: 2025-03-05 14:00:00 +0900
categories:
  - Paper Review
tags:
  - NLP
  - EMNLP 2024
  - Persona-based Dialogue
---

요약: 이 연구는 양자화된 대형 언어 모델에서의 성격 특성 일관성을 탐구하며, 역할 놀이 시나리오에서 다중 상호작용 동안 할당된 성격의 안정성을 평가합니다. 성격 불일치를 해결하기 위한 비모수적 방법인 Think2를 제안하고, QRPDA의 일관성을 유지하는 데 효과적임을 입증합니다.

---

# 1 Introduction

- 역할 놀이 대화 에이전트(RPDA)는 주어진 페르소나를 가진 대형 언어 모델(LLM)임.
- 페르소나는 교사, 유명 캐릭터, 역사적 인물 등 다양한 그룹을 대표할 수 있음.
- RPDA의 행동을 역할 놀이 관점에서 설명할 경우, 의인화의 함정을 피하고 LLM의 행동을 조사할 수 있는 개념적 틀을 제공함.
- RPDA는 최근 학계와 산업에서 주목받고 있으며, 감정적 동반자, 인터랙티브 게임, 개인화된 보조기구 등에 활용되고 있음.
- RPDA의 성격 특성의 일관성을 이해하는 것은 사용자 상호작용의 예측 가능성과 신뢰성을 안정하는 데 중요함.
- 개인 정보 보호 문제로 인해 로컬 배포된 RPDA의 인기가 증가하고 있으며, 데이터 전송을 최소화할 수 있음.
- 자원 제약 때문에 최적화 접근법(예: 양자화)이 필요함.
- 여러 연구가 LLM의 성격을 다루었지만, 로컬에서 배포되는 RPDA의 양자화가 행동에 미치는 영향을 조사한 연구는 없음.
- 본 연구는 양자화된 버전의 LLM에서 구축된 RPDA 성격(QRPDA)의 일관성을 조사함.
- 다음과 같은 연구 질문(RQ)을 설정함:
  - RQ1: LLM의 양자화가 QRPDAs의 성격 일관성에 어떤 영향을 미치는가?
  - RQ2: QRPDAs의 성격 일관성을 개선할 수 있는 전략은 무엇인가?
  - RQ3: 로컬 배포된 QRPDA를 위한 최적의 모델 크기, 유형, 양자화 조합은 무엇인가?
- 다양한 LLM과 양자화 수준을 사용한 실험을 설계하여 RQ를 해결함.
- 연구 결과, 양자화가 성격 일관성을 감소시키며, 대화 중에 할당된 특성을 유지하는 데 도전 과제가 있다는 것을 나타냄.
- 성격 변화를 해결하기 위해 "Think2"라는 비모수적 접근법을 제안하며, 이는 양자화된 대화 에이전트의 일관성을 유지하는 데 좋은 결과를 보여줌.

---

# 2 Related Work

- **개인성 척도**:
  - 개인성 특성 평가를 위한 인기 있는 프레임워크: Big Five 모델 (Fiske, 1949)
  - 구성 요소: 개방성, 성실성, 외향성, 친화성, 신경증 (OCEAN)
  - 다양한 평가 도구: Big Five Inventory (BFI) 예 (Fossati et al., 2011)
    - 자기 보고 척도로 44개 항목, 5점 리커트 척도 사용
- **LLM의 심리적 평가 방법**:
  - 자기 보고 방식 (Frisch and Giulianelli, 2024) 또는 
  - 선택 질문 (Jiang et al., 2023b), 인터뷰 과정 (Wang et al., 2023a) 활용
  - PsychoBench를 통한 포괄적 평가 (Huang et al., 2024)
- **인격 분류를 위한 평가**:
  - 인격 특성 할당에서 캐릭터 할당으로 이동할 때 더 세부적인 평가 필요 (언어 평가, 어휘 일관성, 대화 정확도 등) (Wang et al., 2023b, 2024)
- **RPDA의 LLM 개인성 평가**:
  - 기본 설정 (Pellert et al., 2023; Huang et al., 2024) 또는 RPDA 설정에서 수행
  - 주로 프롬프트 (Wang et al., 2023b; Jiang et al., 2024; Wang et al., 2023a) 및 인-컨텍스트 학습 (Mao et al., 2024)을 통해 인격 할당
  - 특정 성격 유형 유도 위한 파라메트릭 접근 시도 (Mao et al., 2024)
- **LLM 개인성 평가 연구의 초점**:
  - 상업적 LLM 및 대형 오픈소스 모델에 더 많은 초점 (Petrov et al., 2024; Jiang et al., 2024)
  - 소형 오픈소스 모델에 대한 연구는 제한적 (La Cava et al., 2024)
- **LLM 상호작용 행동 연구**:
  - Frisch et al.는 협업 스토리텔링을 통해 LLM 행동 탐구, 하지만 페르소나 두 개와 한 번의 상호작용만으로 한정됨 (Frisch and Giulianelli, 2024)
  - Noh et al.는 게임 에이전트 내에서의 상호작용 조사, 그러나 일반 상호작용에 대한 특정 초점 없음 (Noh and Chang, 2024)
  - Ma et al.의 연구는 상호작용 중 할당된 개인성의 불일치 강조, QRPDA에서의 개인성 일관성을 유지하기 위한 더 포괄적인 연구 필요성 부각 (Ma et al., 2023)

---

# 3 Methodology

<img width="813" alt="image" src="https://github.com/user-attachments/assets/93206209-d904-4c1d-bfdf-77ef888bda64" />

- 모델 양자화가 QRPDA 배치에 미치는 영향을 탐색하기 위해 일련의 실험 설계
- 양자화된 모델과 16비트 부동소수점(FP16) 모델 간의 성격 특성 일관성 평가
- 양자화된 모델의 성격 유지 및 변화 관찰

## 3.1 양자화된 온디바이스 LLMs
- 평가를 위해 선택한 4개의 양자화된 온디바이스 LLMs:
  - LLaMA3 8B Instruct
  - Mistral 7B Instruct v0.3
  - Gemma 7B Instruct v1.1
  - Gemma2 9B Instruct
- 7B 매개변수를 중심으로 모델 선정 (메모리 및 계산 자원에 적합)
- 다양한 양자화 수준(FP16, Q8_0, Q4_0)에서 평가
- Q8_0/Q4_0로 메모리 요구량을 1/2 및 1/4로 감소

## 3.2 RPDA 구축

<img width="322" alt="image" src="https://github.com/user-attachments/assets/a693b641-5c0f-4986-8afa-97e7aaebc29b" />

- 시스템 프롬프트를 통해 LLM에 성격 특성 할당
- Big Five 성격 모델(OCEAN)의 다섯 가지 성격 차원 활용
- 32개의 이진 성격 조합으로 실험 확장
- 초기화된 성격을 다섯 개의 이진 인덱스로 표현
- 각 성격 쌍 비교를 통해 성격 변화 관찰

## 3.3 다중 턴 상호작용

<img width="319" alt="image" src="https://github.com/user-attachments/assets/ac1fcacc-463a-47e6-a1c1-e3d0a5ad4391" />

<img width="319" alt="image" src="https://github.com/user-attachments/assets/5ee55015-74de-4cc2-8656-61fd14700aee" />

- RPDA 쌍이 반복적인 대화를 통해 자연스러운 다중 턴 상호작용을 시뮬레이션
- 각 턴에서 개인 스토리를 교환하여 정보 접근 및 연속성 유지
- 매 턴마다 BFI 자가 평가를 통해 성격 특성 변화 및 일관성 추적

## 3.4 서사 언어적 특성
- 상호작용 후 OCEAN 점수 및 서사를 수집하여 언어적 특성 분석
- LIWC 및 임베딩(EMBD) 방법을 활용한 암묵적 성격 분석
- LIWC의 한계를 보완하기 위해 EMBD 접근 방식 사용

## 3.5 Think2: 성격 특성 강화
- 기본 접근 방식으로 RPDAs가 초기 성격만을 활용하여 대화 수행
- 상호작용의 진행에 따라 성격 특성이 일관성을 잃을 수 있음
- "Think2" 접근 방식 제안: 내러티브 출력 전에 RPDAs가 성격을 반영하도록 유도
- 성격의 유사성을 강화하여 다중 턴 상호작용 동안의 일관성 유지

---

# 4 Experimental Results

- **실험 프레임워크**: Ollama 프레임워크를 사용하여 LLMs 배포
- **선택한 모델**: 
  - LLaMA3 8B Instruct
  - Mistral 7B Instruct v0.3
  - Gemma 7B Instruct v1.1
  - Gemma2 9B Instruct
- **평가 방법**:
  - 세 가지 목표 양자화 수준: FP16, Q8_0, Q4_0
  - 16 쌍의 상반된 성격을 가진 모델 
  - 각 쌍에 대해 20턴의 상호작용 수행, 15회 반복 실험

## 4.1 OCEAN 점수 시각화

<img width="653" alt="image" src="https://github.com/user-attachments/assets/c478ca76-bef3-4141-981b-77029814f801" />

- 레이더 플롯 생성하여 OCEAN 점수 분석
- 초기 및 20턴 상호작용 후 OCEAN 점수 비교
  - 기본 방법 사용 시 성격 점수가 수렴하는 경향
  - Think2 방법은 안정적인 성격 특성 유지, 효과성 입증

## 4.2 언어적 특징에 대한 회귀 분석

<img width="653" alt="image" src="https://github.com/user-attachments/assets/22375109-8d1e-4a1e-aea7-ca7dbdfc1b61" />

- Gemma2 9B Instruct 모델의 비교 분석 결과 제공
- LIWC 및 EMBD 특징 사용
- 기본 방법에서 교차 검증 정확도가 감소
- Think2 방법이 교차 검증 정확도를 유지하며 성격 일관성 보장

## 4.3 상관 관계 분석

<img width="653" alt="image" src="https://github.com/user-attachments/assets/9f3c08c0-b8e6-4082-8624-0a84633b102d" />

- Pearson 상관 분석 결과 제시
- OCEAN 점수와 EMBD 언어적 특징 간 상관 관계를 계산
- 기본 방법 사용 시 상관 관계의 감소 관찰
- Think2 방법은 상관 관계 손실을 효과적으로 완화, 성격 일관성 유지 강화

## 4.4 논의
- 양자화가 LLM의 성격 일관성 저하를 유도한다는 결과
- 높은 양자화 수준에서 성격 특성의 안정성이 감소
- Q8_0은 효율성과 성격 일관성의 균형을 맞추기에 적합한 옵션
- 양자화 전략은 특정 모델에 맞춰 조정 필요

---

# 5 Conclusions

- 저희 연구에서는 양자화된 LLMs에서 생성된 RPDA의 실험을 통해, 높은 양자화 수준에서 개인성 일관성이 감소한다는 사실을 발견했습니다.
- 이 문제를 효과적으로 완화하기 위해 비모수적 방법인 Think2를 제안하였습니다.
- Think2는 상호작용 간의 안정성을 유지하는 데 효과적입니다.
- Gemma2(Q4_0)와 LLaMA3(Q8_0)는 개인성 특성을 보존하는 최적의 선택으로 부각되었습니다.
- 다각적인 분석 프레임워크를 통해 Think2가 자원 제약이 있을 때 디바이스에서 QRPDA의 신뢰성을 향상시킬 수 있는 잠재력을 보여주었습니다.

---

# 6 Limitations

- **사용된 방법론**: 
  - Big Five Inventory (BFI)만을 이용한 성격 평가 
  - 특정 LLM 집단 및 양자화 수준에 한정

- **미탐구 분야**:
  - 추가 성격 모델 조사 (HEXACO, Myers-Briggs 등)
  - 더 다양한 LLM 조사 (소형 모델 및 하위 억 개 매개변수 모델 포함)
  - 현재 연구에서 다루지 않은 다양한 양자화 기법 탐색

- **다국어 연구**: 
  - 연구가 영어에만 국한됨; 다른 언어로의 연구 확장이 필요

- **소형 LLM의 필요성**: 
  - Phi-3, Qwen2, OpenELM 등 소형 모델을 조사의 필요성 인식
  - 자원 제약 환경에서의 활용 가능성

- **다중 양식 LLM의 조사**: 
  - 텍스트, 이미지, 오디오 등 다양한 입력 유형 통합 가능성
  - 사용자 상호작용 이해 및 응답 향상

- **양자화 방법**: 
  - GGUF 양자화 방법에만 국한; 다른 양자화 기법 (AWQ, GPTQ 등) 탐색 필요

- **다양한 상호작용 시나리오**: 
  - 사용자 인구 통계 및 다양한 모드의 상호작용 포함 필요
  - 연구 결과의 강건성 검증 촉진

- **미래 연구 방향**: 
  - 연구의 다각화 및 보편적인 적용 가능성을 위한 노력이 필요
  - 사용자 경험 향상 및 AI 기술의 책임 있는 개발 도모.

---

# 독자 의견

- 본 연구는 5개의 성격모델만으로만 진행했기 때문에 매우 한정적임, 또한 OCEAN score는 자체 모델이 평가하기 때문에 신뢰성에 대한 한계도 존재함.
- 성격 정보를 매번 프롬프트에 넣는 방식으로 일관성을 향상시키는 아주 간단한 방법으로 문제를 해결함. 하지만 이는 일종의 오버헤드가 될 수 있음.
- OCEAN score와 Linguistic feature 간의 Correlation을 보여주어 논문에 대해 신뢰성이 향상됨.
- 추후 양자화에 따른 성능 하락을 방지하기 위한 방안을 연구할 필요가 있음.
