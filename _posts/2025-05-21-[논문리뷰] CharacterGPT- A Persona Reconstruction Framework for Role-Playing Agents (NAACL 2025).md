---
title: "[논문리뷰] CharacterGPT: A Persona Reconstruction Framework for Role-Playing Agents (NAACL 2025)"
date: 2025-05-21 21:58:40 +0900
categories:
  - Paper Review
tags:
  - NAACL 2025
  - Persona-based Dialogue
---

CharacterGPT는 챕터별 소설 요약에서 인물 특성을 점진적으로 추출해 캐릭터 페르소나를 동적으로 재구성함으로써 일관된 역할 수행을 돕는 프레임워크이다.

---

# 1 Introduction

- 대규모 언어 모델(LLMs)의 급속한 발전은 다양한 AI 시스템의 핵심 모듈로 자리잡으면서 여러 응용 분야에 활용되고 있다 (OpenAI, 2023a,c; Anthropic, 2023; Google, 2024; DeepSeek-AI et al., 2024, 2025).
- 문서 기반 정보 검색을 위한 도구인 Assistants API (OpenAI, 2023b)가 도입되면서 특히 역할극 에이전트(Role-Playing Agents, RPA) 분야에서 LLM의 잠재력이 주목받고 있다 (Kim et al., 2019; Yu et al., 2023; Jiang et al., 2023; Park et al., 2023; Wang et al., 2023b; Zhang et al., 2024; Kong et al., 2024; Wang et al., 2025).
- 그러나 문서만을 입력으로 사용하는 RPA는 주요 성격 특성이나 배경 지식이 누락되어 일관성 없는 정보 추출 문제를 겪는다(Sadeq et al., 2024).
- 예를 들어, 그림 1에서 볼 수 있듯이 소설 *God’s Blessing on This Wonderful World!*의 비구조화된 Wiki 문서를 입력했을 때 Assistants API는 부정확한 답변을 자주 생성하지만, 287개의 구조화된 캐릭터 특성을 활용하면 보다 신뢰할 수 있는 역할별 답변을 제공한다.
- 본 논문에서는 CharacterGPT라는 새로운 프레임워크를 제안하여, 구조화된 인물 재구성(Structured Persona Reconstruction) 과정을 도입해 이러한 문제를 해결한다.
- 인지 기억 모델에서 영감을 받은 Character Persona Training (CPT)은 소설의 장별 요약에서 특성을 추출하여 인물 페르소나를 점진적으로 업데이트하는 방법으로, 인간의 기억이 정보를 스키마로 통합하는 방식에 착안했다 (van Kesteren and Meeter, 2020).
- CPT는 인물 분석 문헌(Forster, 1927; Reams, 2015)을 기반으로 다음 8가지 필수 특성을 식별한다:
  - 성격(personality)
  - 신체 묘사(physical description)
  - 동기(motivations)
  - 배경(backstory)
  - 감정(emotions)
  - 관계(relationships)
  - 성장과 변화(growth and change)
  - 갈등(conflict)
- 각 소설 장별 요약에서 이 특성들을 추출하여 인물의 페르소나 문서에 시간순으로 추가함으로써 캐릭터의 발전을 반영한다.
- 추출된 특성들은 서로 분리하여 업데이트되어 통합되지 않고 독립성을 유지한다.
- 이렇게 재구성된 페르소나 문서는 Assistants API에 입력되어 캐릭터의 진화하는 정체성에 기반한 보다 문맥상 적절하고 일관된 답변 생성이 가능해진다.
- 이 프레임워크는 전통적인 문서 기반 검색 방식에서 발생하는 정보 손실과 계산 비용을 최소화하며, 인물 특성을 체계적으로 정리하고 시간에 따라 업데이트한다.
- 또한, CharacterGPT는 소설 내 특정 순간의 페르소나를 생성할 수 있어(예: 악마왕에 맞서기 직전 영웅의 생각 체험) 사용자와 캐릭터 간 상호작용의 시점을 조절할 수 있게 한다.
- CharacterGPT의 효과는 인간 평가를 통해 검증되었으며, 각 캐릭터는 빅 파이브 성격 검사(Big Five Inventory, BFI) (Barrick and Mount, 1991)를 통해 성격 일관성을 평가받았다.
- 또한, 캐릭터들은 단편 내러티브를 생성하여 창의적 능력을 시험받았고, 7명의 크라우드워커가 6가지 지표를 5점 척도로 평가하였다.
- 평가 결과 CharacterGPT는 기존 문서 기반 시스템에 비해 페르소나 일관성, 제어 가능성, 역할별 지식 측면에서 유의미한 개선을 보였다.

---

# 2 Proposed Method

- CharacterGPT의 목표  
  - 페르소나 문서 $$D$$와 추론 프롬프트 $$P_f$$를 입력으로 받아 캐릭터 반응 $$R$$을 생성하는 페르소나 기반 어시스턴트 $$f$$ 구축  
  - 페르소나 문서 $$D = \{s_1, s_2, \dots, s_N\}$$는 $$N$$개의 문장으로 구성  
  - 기존 Assistants API는 전체 문서를 입력으로 사용하나, 인물 특성 반영이 불충분하여 부자연스러움 문제 존재  
  - 이를 해결하기 위해 페르소나 문서를 정제된 버전 $$D_r$$으로 재구성하고 응답을  
    $$
    R = f(D_r, P_f)
    $$  
    로 정의

## 2.1 Preliminaries

- 캐릭터 특성 정의 (Forster, 1927; Reams, 2015)  
  1. Personality (성격): 용기, 내성, 재치 등 핵심 성격 특성  
  2. Physical Description (외모): 캐릭터의 신체적 모습  
  3. Motivations (동기): 행동을 이끄는 목표 및 욕망  
  4. Backstory (배경): 성격과 동기를 형성하는 과거 이야기  
  5. Emotions (감정): 반응에 영향을 미치는 감정 범위  
  6. Relationships (관계): 다른 캐릭터와의 상호작용  
  7. Growth and Change (성장과 변화): 이야기 진행 중 발전 과정  
  8. Conflict (갈등): 내적/외적 갈등 요소

- 사용된 페르소나 문서  
  - 네 명의 캐릭터 (Megumin, Anya Forger, Frieren, Hitori Gotoh) 대상  
  - Namuwiki에서 캐릭터 정보, 줄거리 요약, 대사 수집  
  - Table 1은 챕터 수, 토큰 통계, 대화 라인 수 등 데이터 요약  

## 2.2 Persona Initialization

- 단순 문서 표본 활용은 특성 추출에 한계 존재  
- 따라서 두 단계 재구성 과정 적용: (i) Initialization, (ii) CPT(Character Persona Training)  
- Initialization 단계에서는 이야기 전개 미반영, 스토리 진행 관련 내용 제거  
- 수집한 캐릭터 정보를 다음 5가지 핵심 특성으로 정리:  
  - Personality, Physical Description, Motivations, Backstory, Relationships  
- 이들을 초기 페르소나 문서로 구성  
  $$
  D_{init} = \{D_{per}, D_{phy}, D_{mot}, D_{back}, D_{Rel}\}
  $$
- 감정, 성장 및 갈등 특성은 이후 CPT 단계에서 다룸

## 2.3 Character Persona Training

- 특성 분류  
  - 인간 지식은 내부 속성(Type A)과 외부 속성(Type B)으로 구분 가능  
  - Type A: 본질적 특성 (Personality, Physical Description, Motivations)  
  - Type B: 환경과의 상호작용에 따른 외부 지식 (Backstory, Emotions, Relationships, Growth and Change, Conflict)  
- CPT 과정에서 Type A 특성은 핵심 내부 속성을 일반화하고, Type B 특성은 역할에 따른 외부 지식으로 누적  
- 학습 단계  
  - 각 에폭(epoch)마다 챕터 요약에서 역할별 특성 추출  
  - 수식  
    $$
    T_t^i = \begin{cases} 
    h\big(g(D_i, P_g), P_h\big), & t \in \text{Type A} \\
    g(D_i, P_g), & \text{otherwise}
    \end{cases}
    $$
  - 여기서 $$i$$는 에폭 인덱스, $$D_i$$는 챕터 요약, $$g$$는 Assistants API (프롬프트 $$P_g$$), $$h$$는 LLM 기반 일반화 함수 (프롬프트 $$P_h$$)  
  - Type A 특성은 내부 속성으로 일반화, Type B 특성은 페르소나 문서에 덧붙여져 누적

## 2.4 CharacterGPT

- CPT를 통해 각 캐릭터의 페르소나를 반복적으로 구축하는 방법 제시  
- 장점  
  1. 페르소나 누적을 이야기 진행과 맞춰 정보 손실과 계산 비용 최소화  
  2. 본 시스템은 각 에폭마다 주인공 페르소나를 저장/갱신하며, 사용자가 특정 이야기 시점에서 캐릭터와 상호작용 가능  
- 최종 페르소나는 초기화 페르소나 $$D_{init}$$, 훈련된 페르소나 $$D_{train}$$, 톤 $$T_v$$를 포함  
  $$
  D_r = D_{init} + D_{train} + T_v
  $$
- 톤 $$T_v$$는 대화 자연스러움에 도움이나, 현재 데이터는 주로 정보 및 줄거리 요약 위주로 수집되어 있어 후속 연구 대상

---

# 3 Experiments

- **3.1 Setup**
  - CharacterGPT를 Assistants API와 GPT-4 Turbo("gpt-4-1106preview") 버전을 활용해 구현.
  - 모델 호환성 검증을 위해 ChatGPT("gpt-3.5-turbo-1106")를 이용한 실험 및 제거 연구(ablation study)도 수행.
  - ChatGPT는 해당 버전에서만 Assistants API의 Retrieval 기능 지원.
  - 일반화 함수 $$h$$는 최대 토큰 길이 $$4096$$과 온도 매개변수 $$0.7$$로 설정됨.

- **3.2 Evaluation Protocols**
  - 주요 연구 질문(RQ)은 두 가지 핵심 과제에 대해 실험:
    1. 캐릭터 페르소나(persona)를 어떻게 더 잘 활용할 수 있는가?
    2. 캐릭터가 새로운 아이디어 생성에 상상력을 어떻게 활용하도록 격려할 수 있는가?
  - RQ1: 페르소나 평가
    - 네 명의 캐릭터를 여러 번 읽은 저자 중 한 명이 분석한 성격 특성과 LLM들이 생성한 특성을 비교.
    - 공정성을 위해 네 캐릭터 각각에 대해 모델 결과 평균 산출.
  - RQ2: 스토리 생성
    - 생성된 스토리를 문법, 일관성, 호감도, 관련성, 복잡도, 창의성 6개 기준으로 평가.
    - 자동 평가 방법들이 개발 중이나, 인간의 평가 선호를 반영하는 지표와 벤치마크가 부족하기에 7명의 크라우드 워커가 인간 평가 수행.
  - 케이스 스터디 진행:
    - 특정 시점에서 캐릭터와의 인터랙션 성능 분석.
    - 역할별 속성(Type A, Type B)이 CPT 과정에서 어떻게 변화하는지 조사.

- **3.3 Results for Persona Evaluation**
  - 네 캐릭터에 대하여 Big Five Inventory (BFI) 테스트 (각 성격 특성별 24개 문항, 총 120개 질문) 실시.
  - 테스트 결과를 성격 하위 요소(facets) 값으로 변환.
  - 예) Agreeableness(호감성)의 하위 요소로 신뢰(Trust), 감정 기반 판단(Tendermindedness), 직설적이지 않음(Straightforwardness), 겸손(Modesty) 등 포함.
  - 모델과 인간 결과 간 차이(gap) 측정: 
    - \# Wins: 인간 예측과 가장 근접한 facet 개수 (클수록 우수).
    - $$\sum \vert d \vert$$: 차이의 절댓값 합 (작을수록 인간과 근접).
  - ChatGPT, GPT-4 모두에 대해 본 방법 적용 시 두 지표 모두 개선됨.
  - 예시: Megumin의 Neuroticism에서 GPT-4 단독은 우울증 가능성을 과대평가한 반면, 본 방법과 인간은 그렇지 않음.

- **3.4 Results for Story Generation**
  - 각 캐릭터에 대해 "주어진 텍스트를 기반으로 참여감 넘치는 미래 에피소드를 상상하고 약 2000단어 분량의 소설로 작성하라"는 프롬프트 제공.
  - 총 32편의 스토리 생성 (캐릭터별 4편).
  - 7명의 크라우드 워커가 5점 Likert 척도로 평가 수행.
  - 인간 평가 기준: 문법, 일관성, 호감도, 관련성, 복잡도, 창의성.
  - 결과:
    - 본 방법 적용 시 6개 항목 모두에서 개선.
    - 특히, 호감도, 복잡도, 창의성에서 유의미한 상승 관찰.
    - GPT-4도 우수한 성능이나, 구조화된 페르소나 활용을 통한 본 방법이 비구조화 입력 대비 인간 선호도 상향에 크게 기여함.

- **3.5 Case Study**
  - **특정 시점 모델링**
    - Megumin 소설의 16개 챕터 별 요약을 훈련 데이터로 활용해 16개의 모델(epoch별) 생성.
    - 특정 시점 캐릭터의 내면과 감정을 생생히 표현 가능.
  - **Ablation Study**
    - CharacterGPT 미적용 시, 캐릭터 페르소나의 세밀한 표현 실패.
    - 예: 수줍음 많고 유창하지 못한 Hitori 캐릭터가 GPT-4 단독에서는 제대로 표현 안 됨.
    - Frieren은 페르소나 일관성 결여, 부자연스러운 대사, 판타지적 오류(예: "magical" 대신 "arcane arts"에 관심 있음이 canon상 맞음) 발생.
    - CharacterGPT 적용 시 캐릭터 페르소나 보존 능력 크게 향상됨.
  - **Type A 및 Type B 페르소나 진화**
    - Type A: Frieren – 처음에는 인간 감정에 무관심했으나 동료들과의 여정에서 점차 공감능력 상승.
    - Type B: Hitori – 혼자 지내던 캐릭터에서 동료들과 친밀한 유대 형성으로 성장.
    - CPT(캐릭터 맞춤 학습) 과정이 캐릭터 성격 및 역할 변화 반영에 효과적임.
  - 실험 결과는 소설 생성, 롤플레이, 복잡한 에이전트 레벨의 활용 가능성을 시사.

---

# 4 Conclusion

- CharacterGPT는 구조화된 캐릭터 특성을 입력으로 활용하여 페르소나 일관성을 향상시키기 위해 고안된 페르소나 기반 어시스턴트임.
- 제안된 프레임워크는 초기화 단계와 학습 단계의 두 가지 주요 단계로 구성됨.
- 초기화 단계에서는 이야기의 진행과 관련된 내용을 제외하고, 캐릭터를 아직 내러티브가 진행되지 않은 상태로 간주함.
- 학습 단계에서는 각 epoch마다 챕터 요약에서 관련 특성을 추출하여 캐릭터 페르소나를 점진적으로 정교화함으로써 소설 전개에 따른 자연스러운 캐릭터 발전을 모방함.
- 제안된 방법은 사람 평가와 사례 연구를 통해 페르소나의 일관성 유지 및 캐릭터 특유의 지식 보존에 효과적임이 입증됨.
- 향후 연구 방향으로는 더 포괄적인 성격 모델을 지원하여 깊은 추론과 의사결정 능력을 가능하게 하는 프레임워크 확장이 포함됨.
- CharacterGPT의 학습 과정은 다음과 같이 표현될 수 있음:  
  $$ \text{Persona}_{epoch+1} = \text{Refine}(\text{Persona}_{epoch}, \text{Traits}_{chapter\_summary}) $$  
  여기서 각 epoch마다 챕터 요약에서 추출된 특성 $$ \text{Traits}_{chapter\_summary} $$이 페르소나 정교화에 반영됨.

---

# Limitations

- **핵심 특성(Key Traits)**  
  - CharacterGPT는 페르소나 일관성과 지식 보존 측면에서 우수한 성능을 보여주지만, 핵심 특성의 선정은 경험적 결과 외에 공식적으로 검증되지 않음  
  - 본 연구에 포함되지 않은 문화적 및 사회적 맥락과 같은 특성이 캐릭터 모델링(예: 캐릭터의 외교적 상황)에 필수적일 수 있음  
  - 이러한 특성의 중요성과 필요성에 대한 추가 탐구가 필요  
  - 음성 및 말투(Voice and Speech Pattern)는 중요한 특성이지만, 본 연구에 사용된 데이터셋은 대화가 충분하지 않아 이 특성의 완전한 탐색이 제한됨  
  - 미래 연구에서는 캐릭터 말투 모델링에 필요한 대화량을 규명하는 데 집중해야 함  

- **추론 능력(Reasoning Ability)**  
  - CharacterGPT는 페르소나 일관성과 지식 활용 측면에서 유의미한 개선을 보였으나, 추론 능력은 충분히 탐색되지 않음  
  - 표 6의 미래 시나리오 상상 및 이야기 작성 과제에서 GPT-4 대비 호감도(Likability), 복잡성(Complexity), 창의성(Creativity) 등의 지표에서 우수했으나, 점수가 4점 미만으로 추론 능력 향상의 여지가 있음  
  - 페르소나 기반 모델의 추론 깊이를 향상시키기 위한 추가 연구가 필요  

- **환각 현상(Hallucinations)**  
  - 대규모 언어 모델의 응답에서 환각 현상에 대한 연구가 진행되고 있으나, 페르소나 기반 과제의 환각 현상은 드물게 다뤄짐  
  - 이는 페르소나 지식이 현실 세계의 사실과 자주 달라 실제와 다른 내용(예: 마법사가 불꽃 마법을 사용하는 경우)을 포함하기 때문임  
  - 각 소설별 비용 효율적인 평가 기준 개발이 어렵고, 앞으로는 페르소나 관련 환각 현상을 다루는 효율적 방법론 개발에 집중해야 함
