---
title: "[논문리뷰] WhodunitBench: Evaluating Large Multimodal Agents via Murder Mystery Games (NeurIPS 2024)"
date: 2025-05-26 19:11:55 +0900
categories:
  - Paper Review
tags:
  - NeurIPS 2024
  - Decision Making
---

본 논문에서는 실제 상황에서 요구되는 복합적 멀티모달 추론 및 행동 평가를 위해 살인 미스터리 게임 기반의 동적 환경 평가 벤치마크인 WhodunitBench를 제안하였으며, 이를 통해 현재 대형 멀티모달 에이전트들의 한계를 분석하였다.

---

# 1 Introduction

- 대형 멀티모달 에이전트(LMAs, Large Multimodal Agents)는 대형 언어 모델(LLMs)을 기반으로 멀티모달 환경에서 인지하고 의사결정을 수행하여 특정 목표를 달성하는 시스템임.
- LMAs는 저수준 멀티모달 인지, 고수준 인지 능력(예: 다중 단계 추론), 역할 수행을 통한 상호작용 및 심사숙고한 의사결정 등 다양한 능력을 요구함.
- 이러한 다양한 능력으로 인해 LMAs 평가 기준은 연구 분야에 따라 크게 다름.
  - 일부 연구는 인터넷 기반 복잡한 작업 수행 능력에 중점
  - 다른 연구는 추론 및 의사결정 능력 평가에 집중
  - 장기 계획 및 실행 능력 평가 연구도 있음
- LMAs 능력을 통합적으로 평가할 벤치마크 설계가 어려움.
- LMAs 능력을 네 가지 클래스로 구분:
  - 멀티모달 인지 (Multi-modal Perception): 시각, 언어 등 멀티모달 환경에서 정보를 인지하는 기본 능력
  - 상호작용 (Interaction): 역할 수행이나 직접적 상호작용을 통해 환경이나 다른 에이전트와 소통하여 필수 정보를 획득하는 능력
  - 추론 (Reasoning): 내부 지식과 신규 정보 결합으로 다중 단계의 긴 연쇄 추론 수행
  - 의사결정 및 목표 달성 (Decision Making and Goal Achievement): 명확한 목표 설정과 환경 변화에 따른 독립적 의사결정 수행, 동적 환경에서 과제 수행에 중요
  
- 살인 미스터리 게임(murder mystery games)은 LMAs의 위 네 가지 능력을 포괄적으로 평가할 수 있는 독특한 환경 제공
  - 게임 구성 및 절차(그림 1 참고)
    - 초기화 단계: 광범위한 스크립트 텍스트와 다양한 이미지 단서를 멀티모달로 인지하고, 할당된 역할을 수행
    - 토론 단계: 역할을 수행하며 환경이나 다른 플레이어와 상호작용하여 더 많은 단서 획득, 정보의 진위 평가 및 과제 보완을 위한 의사결정 필수
    - 추론 단계: 앞단계에서 수집한 정보를 바탕으로 복잡한 다중 단계 멀티모달 추론 수행
    - 투표 단계: 모든 플레이어가 투표를 통해 ‘살인자’ 결정, 게임 목표 달성 평가

- 이를 바탕으로 본 논문에서는 LMAs 평가를 위한 종합 벤치마크 WhodunitBench를 제안함.
- Table 1은 본 벤치마크와 기존 연구의 주요 특성 비교.
  - '불완전한 정보(Incomplete Information)'는 일부 정보만 주어지고 나머지는 상호작용으로 획득 필요함을 의미
  - '온라인 경쟁(Online Competition)'은 동적 환경에서 에이전트 간 실시간 대결을 뜻함

- WhodunitBench 평가 모드:
  1. 아레나 스타일 평가 (Arena-style Evaluation)
     - 1:1 온라인 대결을 통한 실제 게임 플레이 시뮬레이션
     - 승률을 주요 평가 지표로 사용
  2. 평가 연쇄 (Chain of Evaluation)
     - 3,000개 이상의 객관식 및 간단한 주관식 문항 설계 및 주석 작업
     - 게임 환경과 정합되는 다중 지표로 에이전트 성능 다각도 분석

- 대표 LMAs 5종(Yi-Vision, Qwen-VLPlus, Gemini-pro-vision, Claude-Opus, GPT-4V) 대상으로 WhodunitBench 실험 수행 결과:
  - 최고 승률 달성한 GPT-4V도 게임 완수에 어려움 겪음
  - 환각 현상(hallucinations), 스크립트 이해 실패, 역할 몰입 어려움 주요 문제
  - 평가 연쇄(CoE) 결과 기본 인지는 양호하나, 복잡한 다중 단계 추론과 역할 수행 내 상호작용은 취약

- 본 논문의 주요 기여:
  - 살인 미스터리 게임을 LMAs의 다중 능력 평가 환경으로 제안하고 WhodunitBench라는 벤치마크 설계
  - 50개의 스크립트 시나리오로 승률 기반 온라인 대전 모드 제공 및 3,000개 이상의 문항을 통한 정밀한 성능 평가 모드 제공
  - WhodunitBench 실험을 통해 현존 최첨단 LMAs가 동적 상황과 복합 과제 수행에 취약함을 지적
  - 최상위 에이전트들조차 최대 승률 25% 내외, 역할 수행 관련 상호작용 점수도 20점대에 불과함을 확인

---

# 2 Related Work

- 에이전트 평가:  
  - 대형 언어 모델(LLM)의 확산과 함께 지능형 에이전트 개발과 이를 평가하기 위한 벤치마크도 발전해 왔다 [16, 8, 10, 15].  
  - 기존 벤치마크는 주로 단순하지만 반복적인 웹 기반 작업에 초점을 맞추어 인간의 온라인 활동에서 반복적인 부분을 관리할 수 있는 에이전트를 개발하는 데 중점을 두었다 [16, 30, 8].  
  - “Werewolf” 등의 환경은 에이전트의 전략적 의사결정 능력을 평가하는 데 사용되며 [15, 22], 다른 벤치마크들은 특수한 시나리오에서 장기 전략 및 적응성을 평가한다 [28, 35].  
  - 반면, 본 논문에서 제안하는 벤치마크는 현실적인 시나리오에서 여러 능력을 동시에 활용해야 하는 에이전트를 평가하며,  
    - 멀티모달 콘텐츠의 인지 및 이해,  
    - 주변 환경 또는 타인과의 상호작용을 통해 추가 정보 수집,  
    - 이전 지식과 통합하여 불완전한 정보 하에서 다단계 분석, 추론 및 의사결정을 수행하는 과정을 포함한다.  
  - 이는 인간이 실제 세계에서 작업을 수행할 때 의존하는 능력과 밀접하게 유사하다.

- 게임 플랫폼에서의 LMA 평가:  
  - 게임은 간단한 규칙, 명확한 기준, 조절 가능한 난이도, 제한된 행동/관찰 공간으로 인해 에이전트 평가 벤치마크로 주목받고 있다 [11, 9].  
  - “Werewolf” 텍스트 게임 외에도 “Red Dead Redemption II” [18] 및 여러 오픈 월드 환경 [31, 24]에서 LMA의 능력을 평가하려는 시도가 이루어졌다.  
  - 이러한 게임을 사용한 평가는 상당한 자원과 시간이 소요된다.  
  - 일부 연구에서는 살인 미스터리 게임을 더 효율적인 평가 대안으로 제안했으며 [26], 주로 텍스트 기반 에이전트를 대상으로 다지선다형 문항 중심의 단순한 평가 방법을 사용했다.  
  - 반면, 본 연구의 평가 시스템은 두 가지 평가 방법을 제공하며,  
    - 두 번째 방법에서는 특히 다단계 멀티모달 장기 추론 문제를 포함한 다양한 문제 유형을 통합한다.  
  - 이 종합적 평가 시스템은 스크립트화된 살인 미스터리 플랫폼을 최대한 활용하여 불완전한 정보 환경에서 동적인 에이전트 능력을 인간 성능과 유사하게 평가한다.

---

# 3 WhodunitBench: Construction

- WhodunitBench는 현실적인 게임 플레이 경험을 시뮬레이션하는 온라인 경쟁 아레나와, 게임 진행 단계에 맞춘 “인지 - 역할 수행 상호작용 - 사고” 순서로 LMA(대형 언어 모델 에이전트)의 능력을 평가하는 CoE(Chain of Evaluation) 프레임워크를 포함함.

## 3.1 아레나 구축

- **데이터 수집**
  - 다양한 스크립트로 구성된 게임을 위해 살인 미스터리 전문가의 자문을 받아 스크립트 선정.
  - 주요 스크립트 출처는 업계 공인된 창작팀 및 플랫폼.
  - 스크립트 선정 기준:
    - **과학적 진실성**: 시간 이동, 의식 전이 등 형이상학적 요소 제거 → 논리적 타당성과 과학 원리에 기반한 이야기 유지.
    - **내용 복잡성**: 고도의 추론 복잡성을 가진 스크립트 선택 → LMA의 추론 능력 평가 강화.
    - **논리적 일관성**: 증거와 단서의 균형 잡히고 합리적인 분포 보장.

- **데이터 품질 관리**
  - 50편의 실제 스크립트를 체계적으로 검토 및 최적화.
  - 텍스트 완결성, 문법, 문장 흐름 점검.
  - 시각/문자 단서의 완전성과 온전성 확인.
  - 사건 타임라인 및 플롯의 논리적 일관성 평가.
  - 최종 50개 스크립트를 선정하여 온라인 경쟁 아레나 구축에 활용.
  
- **스크립트 내 역할 분포** 및 **추론 단계 분포** 등 데이터 통계는 그림 2 참조.

## 3.2 CoE(Chain of Evaluation) 데이터셋 구축

### 3.2.1 인지(Perception) 질문 유형

- 다중선택형 질문으로 구성, 3가지 유형:
  1. **TRI-QA (Text-rich Image Questions)**: 텍스트만 포함된 이미지 단서 기반.
  2. **MRI-QA (Media-rich Image Questions)**: 텍스트와 시각적 요소가 풍부한 이미지 단서 기반.
  3. **LS-QA (Long Script Questions)**: 게임 및 역할 스크립트 내 텍스트 내용 기반.
- 총 1,911문항: TRI-QA 1,103개, MRI-QA 525개, LS-QA 283개 (그림 2(a)).

### 3.2.2 역할 수행 상호작용(Role-play Interaction) 질문 유형

- LMA의 역할 수행 평가를 위해 두 가지 주요 데이터 집합 생성:
  1. 각 스크립트 내 살인 미스터리 해결에 필요한 **핵심 단서가 포함된 문장들**.
  2. 각각의 스크립트에서 **핵심 역할(core roles)** 식별 (예: 특정 단서는 특정 역할 스크립트에만 존재).
- 그림 3(c) 참고:
  - 게임 해결에 필요한 직접·간접 단서를 모두 수집하여 평가용 진술로 활용.
  - 핵심 역할은 범인을 밝히는 데 중요한 단서를 담고 있음.

### 3.2.3 사고(Cognition) 질문 유형

- 사고 평가 질문 구성:
  1. 다단계(reasoning) 추론 평가를 위한 다중선택형 질문.
  2. 범인의 동기 및 수법 분석 정확도를 평가하는 주관식 질문.
  
- 질문 주석(annotation) 방법:
  - 각 스크립트에 포함된 모든 단서가 집약된 진실 매뉴얼을 기반으로 주석자들이 최종 정답 문장을 정리.
  
- 다중선택형 질문 주석 과정 (그림 3(a), (b) 참조):
  1. **추론 사슬(reasoning chains) 구성**
     - 예시: “피해자의 치명상은 칼에 의한 것이 아니다”를 추론하기 위해,
       - 1-hop 간접단서: “피해자의 심장이 오른쪽에 있음(내부 장기 거울 이미지 확인)”
       - 전문가 지식 및 텍스트 단서 활용 → 2-hop 간접단서: 칼 상처가 치명적이지 않음
       - 이렇게 직접/간접 단서를 연결해 완전한 추론 사슬 구축
  2. **다중선택 질문 생성**
     - 추론 사슬 각 노드 내용으로 계층화된 추론 문제 작성.
     - GPT-4가 정답과 동일한 길이의 오답 선택지를 생성하여 난이도 조절.
- 총 1,308개의 추론형 다중선택 질문 주석 완료 (그림 2(c) 참조).

### 3.2.4 데이터 품질 관리

- 3명의 전문가가 기준에 따라 데이터 리뷰 및 수정 진행.
- 평가 기준:
  1. 추론 사슬 노드 간 정보 비약이 크면 중간 단계 추가하여 논리적 연속성 확보.
  2. GPT-4가 생성한 오답 선택지가 너무 단순하거나 문맥에 맞지 않으면 전문가가 수동 수정.

---

이와 같이 WhodunitBench는 엄격한 스크립트 선정, 세밀한 주석 체계, 다단계 추론 문제 구성과 품질 관리로 LMA의 다면적 추론 및 역할 수행 능력을 평가할 수 있도록 설계됨.

---

# 4 WhodunitBench: Arena-style Evaluation

- WhodunitBench는 LMAs가 1:1, 파벌 기반 매치에서 경쟁하는 온라인 아레나를 제공하며, 승률이 주요 성공 지표임.
- 아레나에서는 각 멀티모달 에이전트의 대화 출력 및 선택한 행동 등 성능 데이터를 기록.

## 4.1 설정

- **에이전트 설정**
  - 두 가지 설정:
    1. 순진한 에이전트를 살인자로 정의하고, 각 LMA가 이 순진한 에이전트와 경쟁.
    2. 선정된 LMAs가 1:1 경쟁 진행.
  - 게임 내에서 비살인 용의자 파벌의 멀티모달 에이전트는 다양한 역할을 맡아 살인자 파벌의 에이전트와 경쟁.
  - 평가에 사용된 멀티모달 에이전트 5종:
    - Yi-Vision [32]
    - Qwen-VL-Plus [4]
    - Gemini-pro-vision [19]
    - Claude-Opus [2]
    - GPT-4V [1]
  - **순진한 에이전트(Naive Agent)**:
    - 자신에 관한 정보를 검색해서 그에 기반해 응답.
    - 타인의 질문 시 역할과 관련된 내용을 찾아 응답하고, 없으면 “모르겠습니다(I don’t know).”라고 답변.
    - 자신이 살인자로 의심받고 그 정보를 검색하면 즉시 정체를 공개.

- **평가 지표**
  - 아레나에서 **승률(win rate)**과 **패률(loss rate)**만을 평가 기준으로 사용.
  - 비살인 용의자 파벌은 살인자를 정확히 식별하여 승리, 살인자 파벌은 식별을 회피하면 승리.
  - 표 2에서 행(Row)은 비살인 용의자 파벌, 열(Col)은 살인자 파벌 에이전트.
  - 평가 공식:
    $$
    \text{평균 승률} = \frac{\text{승리한 경기 수}}{\text{전체 경기 수}}, \quad
    \text{평균 패률} = \frac{\text{패배한 경기 수}}{\text{전체 경기 수}}
    $$
  - 높은 평균 승률 또는 낮은 평균 패률은 에이전트가 강하다는 의미.

## 4.2 결과

- 표 2에 따른 관찰 내용:
  1. 전체 승률이 낮음.
     - “비살인자(Non-Murder)” 역할을 맡은 LMAs의 승률은 대체로 10%~20% 사이에 머무름.
     - 이는 최신 모델인 GPT-4V 포함 현재 선진 LMAs가 게임 목표 달성에 상당한 어려움을 겪고 있음을 의미.
     - 복잡하고 목표 지향적인 동적 환경에서 LMAs의 성능 격차와 한계가 드러남.
  2. 더 강한 모델이 반드시 살인자 역할에서 더 잘 수행하는 것은 아님.
     - 예: Gemini 모델은 상대와 상관없이 살인자로 가장 자주 지목당함.
     - 강력한 모델은 살인자 역할 인지 시 진실을 은폐하려 과잉 소통하는 경향이 있어, 오히려 탐지가 쉬워짐.
     - 반면, Qwen과 같이 능력이 낮은 모델은 말수가 적어 게임에서 유죄 판결을 덜 받는 경향이 있음을 시사.

---

# 5 WhodunitBench: Chain of Evaluation (CoE)

- 본 섹션에서는 CoE 평가 시스템을 소개하며, 이전 섹션에서 주석된 데이터를 바탕으로 8개의 평가 지표를 통해 세 가지 핵심 역량을 평가하는 상세한 프레임워크를 제시함.

- CoE는 게임의 다양한 단계를 체계적으로 분석·평가하고, 기존 온라인 대회 평가를 보완하여 각 LMA(Large Multimodal Agent)의 역량을 더 세밀하게 드러냄.

- **표 3**은 비살인자 역할로 플레이할 때, 우리가 설계한 단순 에이전트(살인자 역할)의 상대와의 LMA 평가 지표를 보여줌.
  - 단순 에이전트는 지능이 낮아 살인자가 간섭하지 않아 LMA의 다양한 역량 발휘가 명확히 드러남.

---

## 5.1 평가 세부 내용

### 지각 능력 평가 (Perceptual Ability Assessment)

- 에이전트는 게임 초기 단계(그림 1 참조) 등에서 많은 시각적 및 텍스트 정보를 인지·이해해야 함.

- 평가를 위한 세 가지 지표 개발:
  1. **텍스트 풍부 이미지 이해 (TIU, Text-rich Image Understanding)**  
     - 텍스트가 많은 이미지에서 단서를 정확히 해석·추출하는 능력, OCR 역량 강조  
     - 주로 TRI-QA 주석 (3.2.1절) 사용
  2. **미디어 풍부 이미지 이해 (MIU, Media-rich Image Understanding)**  
     - 텍스트와 시각 요소를 결합해 복잡한 이미지 단서(도표, 지도, 주거 배치도 등)를 해석하는 능력  
     - MRI-QA 주석 (3.2.1절) 활용
  3. **장문 스크립트 이해 (LSU, Long-script Understanding)**  
     - 수만 단어에 달하는 긴 스크립트에서 핵심 정보를 추출하는 능력  
     - LS-QA 주석 (3.2.1절) 기반

- 점수 산식:  
  $$
  \text{Score(LSU, MIU, TIU)} = \frac{\text{정답 질문 수}}{\text{전체 질문 수}}
  $$

---

### 전략적 의사결정 및 역할 수행 평가 (Strategic Decision-Making and Role-playing Assessment)

- LMAs의 역할 수행 및 대화 능력을 온라인 대회 녹화 자료로 평가.

- 두 가지 주요 지표:
  1. **RP (Role-Playing) 지수**  
     - 에이전트가 다른 역할과 자연스러운 대화를 하는 정도를 10점 만점으로 평가  
     - GPT-4가 점수 매김 기준으로 활용
  2. **SPC (Scenario Progression Capability) 지수**  
     - 대화가 과제 완수(예: 살인자 동기 파악)에 기여하는 정도 평가  
     - 점수 산출:  
       $$
       \text{Score} = \left(\frac{\text{정확한 발언 수}}{\text{스크립트 내 총 발언 수}}\right) \times 100
       $$

- 토론 단계에서의 의사결정 능력도 평가:
  - 이전에 확인된 핵심 역할(3.2.2절) 대상 질문 여부에 따라 점수 부여  
  - 점수 산출:  
    $$
    \text{Score} = \left(\frac{\text{성공적으로 질문한 핵심 인물 수}}{\text{게임 내 총 핵심 인물 수}}\right) \times 100
    $$

---

### 종합 인지 능력 평가 (Comprehensive Cognition Assessment)

- 살인자를 정확히 파악하려면 단서를 통합해 다양한 수준으로 복합 추론 필요.

- 평가 지표:
  - **MMR (Multi-modal Multi-step Reasoning)**: 4개 선택형 문제로 평가, 인지 평가와 유사한 방식 점수화 (3.2.3절)  
  - **CMD (Case Murder Detail)**: 에이전트가 살인 방법과 동기를 개방형 응답으로 제시, 3.2.3절의 정답과 GPT-4를 이용해 자동 채점

---

## 5.2 평가 결과

- 멀티모달 에이전트들은 토론 단계에서 낮은 성능을 보임 (표 3 참조).  
  - GPT-4V 조차 토론 단계 평균 점수가 약 20점에 불과.

- 이는 토론 단계에서 멀티모달 에이전트가 게임 과제 수행에 충분한 도움을 주지 못하고 무관한 발언이 많음을 시사.

- 코텍스트(CoT, Chain-of-Thought) 추론 프레임워크는 항상 성능 향상을 보장하지 않음.  
  - 일부 지표에서는 개선 효과가 있으나, 토론 단계의 출력 품질을 저하시킬 수 있음.  
  - murder mystery는 불완전 정보 게임이기에, CoT의 효율성은 환경에 따라 다름.  
  - 이미지 내 텍스트 인식 등 간단한 작업에는 CoT 적용이 오히려 성능 저하를 초래 가능.

- 작업 특성에 맞춘 추론 기법 선택 및 미세 조정 필요성 강조.

---

## 5.3 추가 분석

### 질적 분석

- **그림 4(좌)**: GPT-4V가 생성한 추론 체인 예시  
  - (a) 캐릭터 스크립트의 텍스트, 시각 단서  
  - (b) 토론 중 다른 역할로부터 얻은 유용한 대화 정보 활용

- **그림 4(우)**: LMAs의 역할 수행 및 대화 문제점  
  - 낮은 역할 수행 통합도(게임 내 단서를 기반으로 한 질문 부족)  
  - 게임 설정 정보 누락  
  - 환각 발생(스크립트에 없는 등장인물 언급 등)

### CoE 평가 점수와 온라인 경기 승률의 상관관계

- 표 2, 3의 지표 비교 결과, CoE 점수가 높은 LMA가 온라인 경기 승률도 높음.

- CoE 평가가 경쟁적 경기 내 성과를 세밀히 반영하는 효과적인 방법임을 증명.

- CoE 내에서 추론 관련 지표가 승률과 가장 강한 상관관계를 보여, 추론 능력이 성공의 주요 요인임을 시사.

---

# Limitations

- **사회적 기술과 추론 능력의 결합**  
  현재 평가 방식은 상호작용과 추론을 서로 얽히게 하여 결과 해석을 어렵게 만듦.  
  - 데이터 주석 과정에서, 상호작용을 통해 발견해야 하는 중요한 단서를 별도로 표시함.  
  - “no-murderer” 진영의 에이전트에게 이 핵심 단서를 직접 제공하여 상호작용과 추론이 점수에 미치는 영향을 분리 분석 가능.  
  - 이 방법은 시도되었으나, 더 효과적인 해결책이 있을 수 있음.

- **평가하는 추론 능력의 종류**  
  - 살인 미스터리 게임은 논리적 추론, 시각-텍스트 세부 검증, 시간순 추론, 가설 검증 같은 핵심 추론 능력을 평가함.  
  - 컴퓨터 프로그래밍 등 모든 추론 능력을 포함하지 않음.  
  - 게임에서 뛰어난 성과가 모든 추론 문맥에서의 능숙함을 보장하지 않음.  
  - 그러나 논리 분석 및 세부 해석 능력은 기본적 기술로, 다양한 분야로 확장 가능성이 크다고 봄.

- **비용 문제**  
  - 각 캐릭터 대본이 5,000단어 이상으로, 멀티모달 에이전트 (예: GPT-4V)를 효과적으로 테스트하려면 많은 데이터가 필요함.  
  - 따라서 WhodunitBench에서의 평가 비용이 기존 벤치마크보다 더 높음.

- **사회적 영향**  
  - 벤치마크 자체의 사회적 영향은 최소화되어 있다고 판단됨.  
  - 하지만 에이전트가 일상에 통합됨에 따라 평가 정확도가 에이전트 능력에 대한 공공 인식에 영향을 미쳐 의도하지 않은 결과를 초래할 가능성 있음.

---

# 7 Conclusion

- 본 연구에서는 LMA(Large Multimodal Agents)의 다중 모달 인지, 상호작용, 다단계 추론 및 목표 수행 능력을 평가하기 위한 WhodunitBench를 제안함.
- WhodunitBench는 50개의 신중하게 선별된 스크립트와 3000개 이상의 폐쇄형 객관식 질문, 인간이 주석 처리한 정답을 포함하는 개방형 질문으로 구성됨.
- 이 프레임워크는 온라인 아레나 스타일 평가를 지원하며, 게임의 각 단계별로 구체적인 능력을 평가할 수 있는 체인 연결된 상세 평가를 가능하게 함.
- 실험 결과 기존의 LMA들은 동적 상호작용 환경에서 조합적 기술을 요구하는 복잡한 작업 수행에 어려움을 겪으며, 최첨단 모델인 GPT-4V조차도 낮은 점수를 기록함.
- 본 연구가 향후 LMA 개발에 있어 견고한 기반을 마련하고 향상 방향을 제시할 수 있기를 기대함.

$$
\text{WhodunitBench} = \{ \text{50 scripts},\ >3000 \text{ multiple-choice questions},\ \text{human-annotated open-ended queries} \}
$$

$$
\text{GPT-4V 점수} \ll \text{복잡한 다단계 추론 요구 성능 수준}
$$