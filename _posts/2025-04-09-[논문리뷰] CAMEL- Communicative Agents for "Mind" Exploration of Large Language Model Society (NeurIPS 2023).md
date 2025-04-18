---
title: "[논문리뷰] CAMEL: Communicative Agents for \"Mind\" Exploration of Large Language Model Society (NeurIPS 2023)"
date: 2025-04-09 13:38:29 +0900
categories:
  - Paper Review
tags:
  - NeurIPS 2023
  - Multi-agent
---

이 논문은 대화형 언어 모델이 복잡한 작업 수행에서 성공하기 위해 인간의 입력에 의존하는 문제를 해결하기 위해, 역할 놀이 기반의 새로운 대화형 에이전트 프레임워크를 제안하고 자율적인 협력 기술을 개발하는 방법을 탐구합니다.

---

# 1 Introduction

- 마빈 민스키의 인용: "지능의 마법 같은 트릭은 없으며, 지능의 힘은 다양한 요소의 조화에서 비롯됨."
- 복잡한 실제 작업 해결은 종종 여러 단계를 요구함.
- 최근의 대화형 대규모 언어 모델(LLMs)의 발전은 복잡한 작업 해결에서 주목할 만한 성과를 보여줌.
- 이런 모델의 성공은 인간의 입력에 크게 의존함: 
  - 사용자들은 의도에 맞는 적절하고 정확한 프롬프트를 제공해야 함.
  - 이는 도전적이고 시간 소모적이며, 때로는 불가능할 수 있음.
- 전문 지식이 부족한 사용자는 적절한 프롬프트를 만드는 데 어려움을 겪음.
- 주요 질문: 인간 개입 없이 작업 완료를 유도할 수 있는 자율 커뮤니케이터 대리인을 만들 수 있을까?
- 이를 해결하기 위해 자율적으로 작업을 완료하는 커뮤니케이터 대리인의 가능성, 능력 및 한계를 조사할 필요가 있음.
- 여러 대리인 간의 상호작용 이해는 인공지능의 미래 예측에 중요함.
- 협력하거나 경쟁하는 대리인의 역학은 AI 시스템 성공에 중요한 역할을 함.
- 본 논문은 자율 협력을 촉진하는 기술 개발 가능성을 탐구함:
  - 각종 도전 과제를 발견함 (예: 역할 전환, 명령 반복, 무한 메세지 루프 등).
  - 인간의 의도와 일치시키는 방법 및 효과적인 협력 방법 탐색 필요.
- 제안하는 새로운 협력 대리인 프레임워크: 역할 놀이.
  - 적은 인간 입력으로 대화 유도.
- 공개된 라이브러리 제공:
  - 다양한 대리인 구현, 잘 구성된 프롬프트 예시, 데이터 탐색기 포함.
- 역할 놀이는 대화형 데이터를 생성하는 확장 가능한 방법을 제공.
- 두 가지 협력 시나리오를 고려하여 대화형 데이터셋 AI Society와 Code 생성.
- Math 및 Science 데이터셋을 수집하여 LLM의 능력 출현 연구.
- Misalignment 데이터셋도 생성하여 자율 대리인 시스템의 잠재적인 리스크 시뮬레이션.
- 수집된 데이터셋은 대화형 모델 연구에 유용함.
- 역할 놀이는 고급 언어 모델 발전을 위한 대화 지시 데이터 생성의 확장 가능한 방법 제공.
- 역할 놀이 프레임워크가 gpt-3.5-turbo보다 더 나은 성능을 보임.
- LLaMA를 이용한 지식 출현 연구 및 HumanEval과 HumanEval+에서 코드 생성 능력 평가.
- 기여 사항:
  1. 자율 협력을 위한 새로운 대리인 프레임워크인 역할 놀이 소개.
  2. 다중 대리인 시스템의 협력 행동 및 능력 연구를 위한 확장 가능한 접근법 제공.
  3. 시뮬레이션한 네 가지 대리인 협력 시나리오를 통해 LLM 훈련의 중요한 출현을 보여줌.
  4. 다양한 대리인 구현, 데이터 생성 파이프라인, 데이터 분석 도구와 데이터셋을 포함하는 오픈소스 라이브러리 제공.

---

# 2 Related Work

- **통신 에이전트**
  - 에이전트 간의 통신은 오랜 연구 주제 [76, 77].
  - 자연어는 에이전트 간 통신의 가장 자연스러운 형태로 간주됨 [97].
  - 상호작용이 가능한 통신 에이전트는 복잡한 문제 해결 능력 보유 [113, 85, 72 등].
  - AI 에이전트 간의 통신은 경쟁적 [115, 108] 또는 협력적 [40, 27] 환경에서 이루어짐.
  - 협력적 AI는 인간 및 다른 AI 시스템과 협력하여 공동 목표를 달성하도록 설계됨 [24, 125].
  - 협력적 AI 시스템은 다른 에이전트의 필요와 능력을 고려하고, 협업 및 조정을 통해 효율성, 의사결정 개선, 복잡한 문제 해결 가능성 증가.

- **지시 LLM 및 프롬프트 엔지니어링**
  - LLM(대규모 언어 모델)은 다양한 텍스트 데이터로 훈련되어 정밀한 텍스트 완성 [12, 22].
  - InstructGPT는 사용자 의도와 일치하지 않는 문제를 지적하며, RLHF를 통해 LLM의 적합성을 개선 제안 [23].
  - 최근 다양한 지시 및 프롬프트 방법 개발: Chain-of-Thought (CoT) [123], zero-shot-CoT [61], ReAct [126].
  - LLM의 성능 향상에 기여하고 있으며, 대화형 AI 에이전트의 행동을 모방 [33].
  - 데이터 세트 생성은 중요한 도전 과제로, 다양한 접근법 존재: 핸드크래프트 지시 인스턴스 [120], 데이터 자동 생성 방법 [67, 57, 19].
  - 마지막으로, 프롬프트 엔지니어링의 품질이 LLM의 성능에 큰 영향을 미침 [91, 12].
  - 본 연구는 역할 놀이를 통해 에이전트가 서로 프롬프트를 제공하는 ‘Inception Prompting’ 방법을 도입.

- **AI 정렬**
  - AI 정렬 분야는 AI 시스템이 설계자의 목표와 가치에 일치하도록 보장하는 것을 목표로 함 [4, 39].
  - ‘로봇 공학의 3원칙’은 AI 정렬의 최초 시도로, 아이작 아시모프의 SF 이야기에 등장 [6].
  - 정렬된 AI 시스템을 개발하는 것은 바람직한 목표를 달성하고, 원치 않는 결과를 피하는 데 중요함.
  - AI 모델이 거짓, 공격적, 조작적 정보를 생산하지 않도록 연구 중 [56].
  - 높은 정렬 수준을 달성하기 위해 복잡한 윤리적, 철학적, 기술적 문제를 다루어야 함.
  - 다양한 역할 놀이 상황을 연구하여 LLM의 정렬성을 탐색하는 실험 진행.

---

# 3 Methodology

- 본 논문에서는 협력적인 환경에서의 커뮤니케이티브 에이전트를 연구
  - 공동의 관심사를 공유하는 에이전트들
- 주로 조력자-사용자 시나리오에 초점
  - 초기 아이디어가 제공됨
- 에이전트들은 이 아이디어를 구체적인 작업으로 개념화
- 작업을 autonomously (자율적으로) 수행하기 위해 대화 진행
- 이 과정에서는 수많은 대화를 통해 최적의 결과 도출하는 과정이 포함됨

---

# 3.1 Role-playing Framework

- **역할 연기 설정**:
  - 인간 사용자가 구현하고자 하는 아이디어 (예: 주식 시장을 위한 거래 봇 개발).
  - AI 어시스턴트 역할: 파이썬 프로그래머.
  - AI 사용자 역할: 주식 트레이더.
  
- **작업 명세화**:
  - 작업 명세자 에이전트가 아이디어를 구체화하여 명확한 작업을 지정.
  - 예시 작업: 감정 분석 도구를 갖춘 거래 봇 개발.

- **작업 지정의 중요성**:
  - 대화형 에이전트는 비전문가가 이해하기 어려운 명확한 작업 프롬프트가 필요.
  - 작업 명세자는 아이디어 구현을 위한 상상력을 향상시키는 역할.

- **역할 할당**:
  - 작업 명세 후, AI 어시스턴트와 사용자 역할을 배정.
  - 시스템 메시지가 각 에이전트에 전달되어 역할이 정의됨.

- **지시 및 솔루션 과정**:
  - AI 사용자가 지시를 제공하고, AI 어시스턴트가 솔루션을 제공.
  - 대화는 다음과 같은 수식으로 나타낼 수 있음:
    - $$M_t = \{(I_0, S_0), ..., (I_t, S_t)\} = \{(I_i, S_i)\}_{i=0}^{t}$$

- **대화 흐름**:
  - AI 사용자(U)가 이전 대화 세트를 기반으로 새로운 지시 $$I_{t+1}$$를 제공.
  - AI 어시스턴트(A)가 새로운 솔루션 $$S_{t+1}$$을 응답.

- **비평가의 역할**:
  - 롤플레잉 프레임워크의 통제성을 향상시키기 위해 비평가 에이전트를 도입.
  - 결정-making을 위해 피드백 및 제안 선택 가능.

- **결론**:
  - AI 사용자와 AI 어시스턴트가 대화를 통해 협업하여 지정된 작업을 완료.
  - 전체 프로세스는 Figure 1에서 시각적으로 요약됨.

---

# 3.2 Inception Prompting

- **프롬프트 공학의 중요성**
  - 역할극에서 프롬프트 공학이 중요함
  - 역할 지정 및 작업 명세를 위해 처음에만 공학적 접근

- **인셉션 프롬프트**
  - 인셉션 프롬프트는 세 가지로 구성됨:
    - 작업 명세 프롬프트: $$PT$$
    - 도우미 시스템 프롬프트: $$PA$$
    - 사용자 시스템 프롬프트: $$PU$$

- **AI 사회 시나리오 예시**
  - 역할과 작업 정보를 포함하여 창의적이고 구체적으로 명세
  - 에이전트가 올바른 협업을 위해 프롬프트 설계가 필수적

- **도우미 시스템 프롬프트($$PA$$)**
  - 역할 고정: 역할 전환 방지
  - 명령 불능 시 설명 제공
  - 일관된 형식으로 응답 요구: "해결책: <YOUR_SOLUTION>"
  - 항상 끝에 "다음 요청"을 첨가하도록 요구

- **사용자 시스템 프롬프트($$PU$$)**
  - 도우미와 대칭적으로 설계
  - 명령을 필요한 입력과 없는 입력으로 구분
  - 작업 완료 시 "<CAMEL_TASK_DONE>" 응답 요구

- **작업 완료 기준**
  - 사용자가 작업 완료를 인정할 때 대화 종료
  - 사용자가 완료를 주장하기 전에는 계속해서 지시 요청

- **구조적 요구사항**
  - 서로의 역할을 명확히 하여 협업을 원활하게 유지
  - 역할과 책임의 이해를 통해 불필요한 대화 루프 방지

- **코드 시나리오**
  - AI 사회 시나리오와 유사하지만 프로그래밍 언어 관련 추가 공학 적용

- **결론**
  - 프롬프트 설계는 자동 협업을 원활히 하며 효과적인 결과 도출에 중요한 요소.

---

# 4 Experiments

- **실험 개요**
  - 다양한 실험을 통해 최종 디자인 선택을 결정함.
  - 에이전트 간 통신을 통해 자율적 협력을 목표로 한 실험 분석.

- **실험 요소**
  - LLM 에이전트로 두 개의 gpt-3.5-turbo 에이전트 사용.
  - Inception Prompts 활용하여 사용자-도우미 협력 시뮬레이션.

- **AI 사회 설정**
  - AI Society 환경에 집중하여 실험 진행.
  - 대화 데이터 수집:
    - CAMEL AI Society 데이터셋.
    - CAMEL Code 데이터셋.
    - 문제-해결 쌍 데이터셋 (CAMEL Math, CAMEL Science).
  - 데이터 품질 분석 및 평가.

- **데이터 생성 프롬프트**
  - AI 사회 도우미 역할 생성 프롬프트:
    - 사용자가 수행할 수 있는 역할을 목록화.
    - 역할 수: <NUM_ROLES>, 알파벳 순으로 정렬.
  - 사용자 역할 생성 프롬프트:
    - 가장 일반적이고 다양한 인터넷 사용자 집단 또는 직업 목록화.
    - 역할 수: <NUM_ROLES>, 알파벳 순으로 정렬.
  - 작업 생성 프롬프트:
    - 도우미가 사용자와 협력하여 수행할 수 있는 작업 목록화.
    - 작업 수: <NUM_TASKS>, 간결하고 창의적으로 작성.

- **확장 가능 접근법**
  - 데이터 매개변수는 LLM 모델을 사용하여 생성, 인간 개입 최소화.
  - 데이터 생성 프롬프트 요약: 그림 3 참조. 

- **미래의 AI 사회**
  - 프레임워크의 잠재적 확장 논의.
  - 미래 AI 사회에서의 위험과 기회 강조.

---

# 4.1 Role-Playing for AI Society

- AI 사회 데이터셋을 만들기 위한 확장 가능한 접근법 개발
- **단계별 접근법**:
  - **역할 생성**: LLM 에이전트에 사용자와 보조자의 가능한 역할을 생성하라는 프롬프트 제공
  - **작업 생성**: 협업을 통해 해결할 수 있는 다양한 작업 생성 요청
  - **작업 세분화**: 생성된 작업을 더 구체화하기 위한 프롬프트 사용
- **결과**:
  - 보조자 역할: 50개
  - 사용자 역할: 50개
  - 각 역할 조합에 대해 10개의 작업 생성 → 총 25,000개의 대화
- **참고**: 생성된 보조자 역할, 사용자 역할 및 Code, Math, Science 데이터셋 생성에 대한 상세 내용은 부록에 수록.

---

# Challenges and Observations

- **역할 전환 (Role Flipping)**: 
  - 보조자와 사용자가 대화 중 역할이 전환되는 문제.
  - 보조자가 사용자의 프롬프트를 따르지 않고 지침이나 명령을 제공할 때 발생.
  - 보조자가 질문을 하지 않아야 역할 전환을 방지할 수 있음.

- **보조자 지침 반복 (Assistant Repeats Instruction)**: 
  - 역할 전환 없이 사용자의 지침을 단순히 반복하는 문제.

- **즉흥적인 응답 (Flake Replies)**: 
  - 보조자가 "I will..."과 같은 응답으로 임시방편적인 메시지를 보내는 경우.
  - 이러한 응답은 작업에 기여하지 않음.

- **무한 메시지 루프 (Infinite Loop of Messages)**: 
  - 사용자가 서로에게 반복적으로 감사하거나 작별 인사를 하며 작업이 진전되지 않는 경우.
  - 경우에 따라 보조자와 사용자가 루프에 갇혀 있다는 것을 인식하지만 이를 끊지 못함.

- **종료 조건 (Termination Conditions)**:
  - 대화의 일관성과 정확성을 보장하기 위해 설정된 조건.
  - 사용자가 3 라운드 동안 보조자에게 지시하지 않으면 대화 종료.
  - 보조자가 사용자에게 지시할 경우 역할 전환을 나타내며 대화 종료.
  - 사용자가 작업 완료를 느끼면 "<CAMEL_TASK_DONE>" 메시지를 통해 대화 종료.
  - 토큰 한도 도달 시 대화 종료.
  - 최대 40개의 메시지 제한 설정하여 비용 조절 및 충분한 대화 보장. 

- **전반적 관찰**: 
  - 협력적 AI 개발의 복잡성 강조.
  - 이러한 도전 과제를 극복하기 위한 지속적인 탐구와 혁신의 필요성 결론.

---

# 5.1 Agent Evaluation

- CAMEL(협력적인 역할극 커뮤니케이션)의 성능을 평가하기 위해 두 가지 유형의 평가를 수행:
  1. 인간 평가
  2. GPT4 평가

- AI Society 데이터셋 및 Code 데이터셋에서 각각 100개의 임무를 무작위로 선택하여 평가 진행.
  
- GPT4 모델을 사용하여 CAMEL 대화 기반 8 솔루션의 내용을 요약하고 통합 최종 솔루션을 제시.
  
- GPT4가 더 많은 토큰 한계를 가지고 있어 요약에 적합함.
  
- 요약은 CAMEL 에이전트의 솔루션 형식을 숨겨 공정한 비교 가능.
  
- 최종 솔루션은 gpt-3.5-turbo 모델의 단일 샷 솔루션과 비교됨.

### 인간 평가
- CAMEL 요약된 에이전트 솔루션과 gpt-3.5-turbo 단일 샷 솔루션을 인간 참가자에게 나란히 제시.
- 솔루션의 정체는 공개되지 않음.
- 참가자들은 어떤 솔루션이 우월한지 또는 동등한지를 투표.
- 총 453개의 응답 수집.
- 인간 평가는 AI Society에 대해서만 수행되며, 코드 평가는 일반적으로 더 어려움.

### GPT4 평가
- GPT4 에이전트를 활용하여 모델 1(CAMEL 에이전트 솔루션)과 모델 2(gpt-3.5-turbo 단일 샷 솔루션)의 효과성 평가.
- GPT4에게 두 솔루션 중 어느 것이 더 나은지 점수 및 결정하도록 프롬프트.

### 결과
- 평가 결과는 표 1에 요약되어 있으며, CAMEL 솔루션이 인간 평가 및 GPT4 평가 모두에서 gpt-3.5-turbo 단일 샷 솔루션보다 큰 차이로 우수함.
  
- 인간 평가와 GPT4 평가 모두 매우 일치함.

| 데이터셋   | 평가 유형        | 동점   | gpt-3.5-turbo 승리 | CAMEL 에이전트 승리 |
|----------|--------------|-------|-----------------|--------------------|
| AI Society | 인간 평가     | 13.3% | 10.4%           | 76.3%              |
|          | GPT4 평가    | 4.0%  | 23.0%           | 73.0%              |
| Code     | GPT4 평가    | 0.0%  | 24.0%           | 76.0%              |

---

# 5.2 GPT4 for ChatBot Evaluation

- 모델: LLaMA 7B를 다양한 데이터셋으로 점진적으로 파인튜닝
- 첫 데이터셋: AI 사회
  - 인간 상호작용 및 사회 역학 학습
- 추가 데이터셋:
  - **코드**: 프로그래밍 논리 및 문법 학습, 실행 가능한 코드 생성
  - **수학**: 복잡한 방정식 해결, 추상 개념 추론, 정확한 계산 수행
  - **과학**: 과학 이론, 경험적 관찰, 실험 방법 이해 확장
- **능력 평가 방식**:
  - 모델 응답 질 평가, 각 분야의 다양한 난이도 질문 집합 기준
  - 평가 항목: AI 사회 관련 20개 작업, 코드 20개, 수학 20개, 과학 60개
- 결과:
  - 데이터셋 추가 시 모델 성능 향상 관찰 (표 2 참조)
  - 다른 분야에서도 성능 향상이 관찰됨 (예: 코드 훈련 시 과학 향상)
  - AI 사회에서 코딩 역할이 코드 관련 대화 향상
- **비교 결과**:
  - AI 사회 vs AI 사회 + 코드 + 수학의 동률: 방정식 해결 능력 미흡
  - AI 사회에서 코드, 수학, 과학으로의 발전 과정은 인간의 다양한 주제 전문 지식 습득 방식과 유사
- **부록**: 샘플 작업 제공
- **표 2**: 모델 1과 모델 2의 성능 비교, 모델 2가 거의 항상 더 우수함
- **HumanEval(+) 평가**:
  - CAMEL 모델: LLaMA-7B가 모든 데이터셋으로 파인튜닝된 모델
  - **결과**: CAMEL이 LLaMA-7B 및 Vicuna-7B 모델보다 뛰어난 성능
  - 데이터셋의 중요성 강조: LLaMA의 코딩 작업 수행 능력 향상

---

# 6 Conclusion

- 자율적인 의사소통 에이전트 간의 협력 가능성 탐색
- "역할 놀이"라는 새로운 협력 에이전트 프레임워크 제안
- 최소한의 인간 개입으로 작업 완료를 위한 자율 협력 촉진
- 철저한 평가 결과, 더 나은 솔루션 도출
- 자율 협력 달성이 어려운 이유:
  - 대화 이탈
  - 역할 전환
  - 종료 조건
- 프레임워크는 다중 에이전트 시스템의 협력 행동 및 능력을 연구하는 확장 가능한 접근법 제공
- 문제 해결을 위한 전략 제안
- 오픈 소스 라이브러리:
  - 다양한 에이전트 구현
  - 데이터 생성 파이프라인
  - 데이터 분석 도구
  - 수집된 데이터셋
- 의사소통 에이전트 및 그 이상에 대한 연구 지원
- 대규모 언어 인공지능 모델 및 협력 AI 시스템의 미래에 대한 가치 있는 통찰 제공

---

# Acknowledgements

- 이 작업은 SDAIA-KAUST 데이터 과학 및 인공지능 센터의 지원을 받았습니다.