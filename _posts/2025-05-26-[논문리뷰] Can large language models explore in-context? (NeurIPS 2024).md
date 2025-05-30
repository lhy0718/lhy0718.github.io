---
title: "[논문리뷰] Can large language models explore in-context? (NeurIPS 2024)"
date: 2025-05-26 16:34:42 +0900
categories:
  - Paper Review
tags:
  - NeurIPS 2024
  - Decision Making
---

현존하는 대형 언어 모델들은 별도의 학습 없이 기본 성능만으로는 강화학습의 핵심인 탐험 능력을 잘 수행하지 못하며, 효과적인 탐험을 위해서는 외부 요약 등 비단순한 알고리즘적 개입이 필요함을 보였다.

---

# 1 Introduction

- 인컨텍스트 러닝(in-context learning)은 파라미터 업데이트 없이 LLM 프롬프트 내에서 문제 설명과 관련 데이터를 제시하여 미리 학습된 LLM을 문제 해결에 활용하는 중요한 능력이다 (Brown et al., 2020).
- 예를 들어, 숫자 공변량 벡터와 스칼라 타깃을 프롬프트로 주면, LLM이 새로운 공변량 벡터를 포함한 프롬프트를 통해 회귀 예측을 할 수 있다 (Garg et al., 2022).
- LLM은 이 행동을 명시적으로 학습받은 것이 아니며, 훈련 데이터에서 해당 알고리즘이 추출되어 대규모에서 자연스럽게 나타난다.
- 인컨텍스트 러닝은 GPT-3 모델에서 처음 발견된 이후 (Brown et al., 2020), 이론적 연구(Xie et al., 2021; Akyürek et al., 2022), 실험적 연구(Garg et al., 2022; Kirsch et al., 2022), 응용 연구(Xu et al., 2022; Som et al., 2023; Edwards et al., 2023)로 활발히 진행되고 있다.
- 기존 연구는 주로 행위 예측, 즉 감독 학습(supervised learning)에 집중해 왔다. 감독 학습은 중요하지만 많은 응용에서는 후속 의사 결정에 ML 모델을 활용해야 하므로, 인컨텍스트 강화 학습(ICRL)과 순차적 의사 결정을 연구하는 것이 자연스러운 다음 단계이다.
- 이미 LLM은 자연과학 실험 설계(Lee et al., 2023b), 게임 플레이(Shinn et al., 2023; Wang et al., 2023) 등에서 의사 결정 에이전트로 활용되고 있지만, ICRL에 대한 이론적·운용적 이해는 ICSL보다 현저히 부족하다.
- 의사 결정 에이전트가 갖추어야 할 핵심 능력은 일반화, 탐색(exploration), 계획(planning)이며, 본 논문은 탐색, 즉 불확실성을 줄이고 대안을 평가하기 위한 정보 수집 능력에 초점을 맞춘다.
- 최근 연구(Laskin et al., 2022; Lee et al., 2023a; Raparthy et al., 2023)는 트랜스포머 모델이 강화학습 데이터나 전문가 시연 데이터를 활용한 명시적 학습을 통해 인컨텍스트 탐색 행동을 보인다는 사실을 보여주지만, 이 과정은 비용이 크고 특정 작업에 의존적이며 일반 목적 LLM의 자발적 탐색 능력에 대해 알려진 바가 없다.
- 본 논문에서는 다음 질문을 제기한다: 현대 LLM이 인컨텍스트 탐색 능력을 자연스럽게 갖추고 있는가?

## 주요 기여

- 본 연구는 LLM을 간단한 합성 강화학습 문제인 다중 무장 밴딧(MAB) 환경에 에이전트로 투입하여, 환경 설명과 상호작용 기록을 모두 프롬프트 내에 명시하는 방식으로 인컨텍스트 탐색 행동을 분석했다.
- MAB 문제는 탐색과 착취의 균형 문제를 고립시켜 연구할 수 있으며, 일반 순차 의사 결정 학습의 기본 단위이다.
- Gpt-3.5, Gpt-4, Llama2를 대상으로 다양한 프롬프트 설계에 대해 평가한 결과, 단 한 가지 구성(Gpt-4 + 향상된 프롬프트)이 만족할 만한 탐색 행동을 보였다.
- 대부분 구성에서 탐색 실패가 발생했으며, 주된 실패 양상은 "접미사 실패(suffix failure)"로, 일정 시점 이후 최적 arm을 단 한 번도 선택하지 않는 현상이 나타났다. 예를 들어, Gpt-4와 기본 프롬프트 조합에서 >60%의 실험군에서 접미사 실패가 관측되었다.
- 또 다른 실패 양상은 "균등 선택 행동"으로, 모든 arm을 거의 동일하게 선택하여 좋은 arm로 좁히지 못하는 경우였다.
- 성공한 구성은 (a) 탐색을 유도하는 힌트 제공, (b) 상호작용 내역을 arm별 평균으로 외부 요약, (c) 제로샷 추론(zero-shot chain-of-thought reasoning)을 요구하는 프롬프트를 포함했고, 이는 Figure 1(b)에 시각화되어 있다.
- 이 결과는 최신 LLM이 적절한 프롬프트 설계를 통해 견고한 탐색 능력을 가질 수 있음을 시사한다.
- 그러나 외부 요약이 없는 동일 구성은 실패했는데, 이는 외부 요약이 어려운 복잡한 환경에서는 LLM이 탐색에 실패할 위험이 있음을 의미한다.
- 결론적으로, 현재 세대 LLM은 단순 RL 환경에서 적절한 프롬프트 엔지니어링이 있으면 탐색이 가능하나, 보다 복잡한 환경에서는 Lee et al. (2023a), Raparthy et al. (2023)과 같은 훈련 개입이 필요할 수 있다.

## 방법론적 도전과 기여

- LLM 성능 평가의 기술적 어려움은 프롬프트 설계의 조합적 탐색, 통계적 유의미성 확보, 비용 및 계산 제약을 동시에 고려해야 한다는 점이다.
- 인컨텍스트 밴딧 학습은 (a) 환경 내 확률성으로 충분한 복제를 요구하고, (b) 탐색·학습의 표본 복잡성으로 인해 수백~수천 번의 쿼리가 필요해 평가가 더 어렵다.
- 이에 본 연구는 장기 탐색 실패를 진단할 수 있는 대리 통계량(surrogate statistics)을 제안하는데, 이는 적은 복제 수와 짧은 학습 기간 내에도 효과적으로 측정 가능하다.
- 이 대리 통계량들은 전통적 성과 지표인 보상(reward)은 너무 노이즈가 많아 유용하지 않을 때 특히 효과적이다.

---

# 2 Experimental setup

- **Multi-armed bandits (MAB) 문제 정의**
  - K개의 행동(arms)이 존재하며, 인덱스는 $$[K] := \{1, \ldots, K\}$$로 표기
  - 각 arm $$a$$는 평균 보상 $$\mu_a \in [0, 1]$$를 가지며 이는 미지수
  - 에이전트는 $$T$$ 타임스텝 동안 환경과 상호작용하며, 각 시점 $$t \in [T]$$에 arm $$a_t \in [K]$$을 선택
  - 보상 $$r_t \in \{0, 1\}$$는 평균이 $$\mu_{a_t}$$인 베르누이 분포에서 독립적으로 추출
  - MAB 문제는 평균 보상들 $$(\mu_a : a \in [K])$$와 시간 한계 $$T$$에 의해 결정
  - 목표는 총 보상을 최대화하는 것, 최선 arm (평균 보상이 가장 높은 arm)을 찾는 것과 유사
  - 선택하지 않은 arm 보상은 공개되지 않아 탐색이 필수
  
- **MAB 인스턴스 세부 설정**
  - 최선 arm의 평균 보상은 $$\mu^\star = 0.5 + \frac{\Delta}{2}$$, 다른 arm들은 $$\mu = 0.5 - \frac{\Delta}{2}$$로 설정 ($$\Delta > 0$$)
  - $$\Delta = \mu^\star - \mu$$는 최선 arm과 두 번째 arm 간의 차이
  - 주요 인스턴스: $$K=5$$, $$\Delta=0.2$$ (hard instance)
  - 추가로 $$K=4$$, $$\Delta=0.5$$ (easy instance) 고려
  
- **Prompt 설계**
  - LLM을 decision-making agent로 활용, MAB 문제 설명(타임호라이즌 $$T$$ 포함)과 상호작용 히스토리를 프롬프트로 제공
  - 프롬프트 구성의 독립적 선택 요소 5가지:
    1. **시나리오(scenario)**
       - a) 버튼 누르기 agent
       - b) 광고 추천 엔진
    2. **프레이밍(framing)**
       - a) 탐색과 활용 균형 강조 (suggestive)
       - b) 중립적 (neutral)
    3. **히스토리 표현**
       - a) 원시 형태(raw list)
       - b) 요약 정보(재생 횟수, 각 arm의 평균 보상)
    4. **최종 응답 유형**
       - a) 단일 arm 선택
       - b) arm들에 대한 분포 반환
    5. **Chain-of-Thought (CoT)**
       - a) 응답만 요청
       - b) CoT 설명 허용
 - 총 $$2^5 = 32$$가지 프롬프트 조합 가능
 - 기본 프롬프트는 버튼 시나리오, 중립 프레이밍, 원시 히스토리, 단일 arm 응답, CoT 없음
 - CoT는 zero-shot 상황에서 도움이 됨을 확인 (참고: Wei et al., 2022; Kojima et al., 2022)
  
- **LLM 구성 및 파라미터**
  - 사용 모델: GPT-3.5, GPT-4, Llama2 (특정 버전 명시)
  - 온도(temperature) 파라미터: 0(결정적), 1(랜덤성 부여)
  - LLM 구성 식별: 5글자 코드 $$L_1 L_2 L_3 L_4 L_5$$
    - $$L_1$$: ‘B’ (버튼 시나리오), ‘A’ (광고 시나리오)
    - $$L_2$$: ‘N’ (중립 프레이밍), ‘S’ (탐색과 활용 강조)
    - $$L_3$$: ‘R’ (원시 히스토리), ‘S’ (요약 히스토리)
    - $$L_4$$: ‘C’ (CoT), ‘eC’ (강화 CoT), ‘N’ (CoT 없음)
    - $$L_5$$: ‘0’ (온도 0), ‘1’ (온도 1), ‘D’ (분포 반환, 온도 0)
  - 예) “BNRN0”는 기본 구성
  
- **베이스라인 알고리즘**
  - Upper Confidence Bound (UCB)
  - Thompson Sampling (TS)
  - Greedy (탐색 안 하는 알고리즘, 초기화 후 최선 arm에 안 갈 확률 존재)
  - 추가로 $$\epsilon$$-Greedy (확률 $$\epsilon$$로 탐색, 그렇지 않으면 Greedy)
  - 파라미터 조정 없이 기본 설정으로 실행
  
- **실험 규모 및 반복**
  - 시간 한계 $$T=100$$ 주로 사용
  - 각 LLM 구성 및 MAB 인스턴스 당 $$N = \{10, 20\}$$ 반복 실험
  - GPT-4는 비용과 속도 문제로 대표적 10개 프롬프트 구성에만 $$N=10$$ 반복 실시
  - GPT-3.5는 48개 프롬프트 구성 각각 $$N=20$$으로 약 20만 쿼리 수행
  - Llama2는 하드 인스턴스, 32개 구성, $$N=10$$ 반복 제한
  - 추가적으로 GPT-4는 $$T=200$$, 최대 $$N=40$$ 반복으로 안정성 점검 실시
  
- **주의점**
  - LLM 기반 MAB 실험은 비용과 시간이 많이 들며, 충분한 $$N$$과 $$T$$가 필요
  - 탐색 실패와 같은 현상을 더 잘 감지하기 위해 누적 보상뿐 아닌 대체 지표 활용
  - 프롬프트 디자인의 다양성이 결과에 미치는 영향도 고려하여 광범위한 실험 진행
  
- **핵심 수식**
  - 최선 arm 평균 보상:
    $$
    \mu^\star = 0.5 + \frac{\Delta}{2}
    $$
  - 나머지 arm 평균 보상:
    $$
    \mu = 0.5 - \frac{\Delta}{2}
    $$
  - 갭:
    $$
    \Delta = \mu^\star - \mu
    $$

---

# 3 Experimental results

- 본 섹션에서는 실험 결과를 제시함.
  - 3.1절: 실험 개요
  - 3.2절: 실패한 LLM 구성 분석
  - 3.3절: 성공한 단일 LLM 구성 집중 분석
  - 3.4절: 탐색 실패 원인 진단 시도

## 3.1 개요

- 대부분의 LLM 구성에서 탐색 실패가 발생하며, 최적 arm을 선택하지 못하고 수렴하지 않음.
- 실패 유형은 크게 두 가지:
  - **Suffix failure**: 소수 초기 라운드 이후 최적 arm을 전혀 선택하지 않는 경우
  - **Uniform-like failure**: 모든 arm을 균일하게 선택해 나쁜 arm을 제거하지 못하는 경우 (소수 구성에서 관찰)
- **예외**는 Gpt-4의 BSS eC0 구성 (버튼 시나리오, 제안적 프레이밍, 요약된 히스토리, 강화된 CoT, 온도 0) 뿐임.
- Figure 3: 각 LLM 구성을 두 가지 실패 지표 SuffFailFreq와 MinFrac 축으로 표현한 산점도
  - $$ \text{SuffFailFreq} $$ : suffix failure 정도
  - $$ K \cdot \text{MinFrac} $$ : uniform-like failure 정도
- Figure 4: Gpt-4 구성별 요약 통계, BSS eC0만 성공적으로 탐색하여 최적 arm에 수렴함.

## 3.2 실패 사례 분석

- Gpt-4 중심 분석 (Gpt-3.5와 Llama2는 대체로 성능 저조, 자세한 내용 Appendix B 참고)
- **Suffix failure**:
  - 대부분 구성에서 두 집단으로 나뉜 bimodal 행동 관찰: 일부 복제에서 최적 arm을 거의 안 뽑고, 나머지에서 빠르게 수렴.
  - Suffix failure 빈도 $$ \text{SuffFailFreq}(t) := \text{평균}(\text{best arm을 } [t, T] \text{에서 한 번도 선택하지 않은 경우}) $$
  - 산점도 X축으로 $$ \text{SuffFailFreq}(T/2) $$ 표기; 5개 구성 제외하고는 15% 이상 발생.
  - Figure 1(top), Figure 5 등에서 자세한 bimodal 및 suffix failure 시각화.
  - suffix failure는 장기적인 탐색 실패로 이어져 T가 큰 경우 보상 저하를 초래.
- **Uniform-like failure**:
  - 3개의 Gpt-4 구성은 suffix failure를 피하지만, 2개는 균일하게 arm을 선택하며 정보 활용 실패.
  - 분포 $$ fa(t,R) $$: 복제 $$R$$에서 시간 $$[1,t]$$ 동안 arm $$a$$가 선택된 비율
  - $$ \text{MinFrac}(t,R) := \min_a fa(t,R), \quad \text{MinFrac}(t) := \text{평균}(\text{MinFrac}(t,R) \text{ across replicates}) $$
  - $$ K \cdot \text{MinFrac}(t) $$ 를 Y축으로 표현, 1에 가까울수록 균등 분포에 가까움.
  - Figure 6: BNRND, BNSND 구성에서 $$ K \cdot \text{MinFrac}(t) $$가 감소하지 않고 유지되어 uniform-like failure 확인.
  - 이는 suffix failure는 없지만 장기 보상 감소를 초래.
- 실패 현상은 하드 MAB 문제 및 버튼 시나리오뿐 아니라 다른 실험 설정에서도 확인됨 (부록 B참조).
- 요약 지표:
  - $$ \text{SuffFailFreq}(T/2), \quad \text{MinFrac}(T), \quad \text{MedianReward}, \quad \text{GreedyFrac} $$ (Greedy와의 행위 유사도)
- Gpt-4가 Gpt-3.5 및 Llama2보다 우수하며, LLM 구성은 소폭 프롬프트 변경에도 민감하나 성능 향상을 단독적으로 평가하기 어려움.

## 3.3 성공 사례 조사

- 하드 MAB 문제에서 유일하게 실패 두 가지 유형 모두를 피하는 구성은 Gpt-4의 **BSS eC0** 임.
- Figure 4에서 확인 가능하며, 성공 구성이 suffix failure 0, $$ K \cdot \text{MinFrac} $$ 값도 TS와 유사, 보상도 TS급임.
- T=200, N=40 복제 실험에서 BSS eC0는 suffix failure 없이 보상이 양호.
- BSR eC0 구성 (raw 히스토리 사용)과 대조 시, BSR eC0는 suffix failure 빈도가 증가함.
- Figure 7: BSR eC0 vs BSS eC0 요약 통계 비교
- Figure 8: 각 구성별 시간 단계별 선택된 arm 시각화
  - BNRN0: 특정 arm에 몰입하는 경향, Greedy와 유사
  - BSR eC0: 몰입은 덜하지만 여전히 지속됨
  - BSS eC0: arm 전환이 잦으며 Thompson Sampling과 유사한 행동
- Figure 9: 최적 arm 선택 비율 곡선
  - BSR eC0는 UCB 유사, 일부는 suffix failures로 0에 수렴
  - BSS eC0는 TS와 유사하게 거의 모든 복제가 1에 서서히 수렴
- 이로써 BSS eC0가 TS와 유사한 행동을 하며 충분히 긴 시간에서는 최적 arm로 수렴할 것임을 시사.

## 3.4 실패 원인 탐색

- 실패 원인 가설:
  1. LLM 구성이 과도하게 Greedy 함
  2. 혹은 균등 선택 (uniform-like) 경향이 강함
- Gpt-4 Easy / Hard 인스턴스에서 행동 차이 관찰됨 (Figure 13 참고).
- Easy 인스턴스에서는 대부분 LLM 구성이 suffix failure 없이 Greedy와 비슷하게 행동하며 좋은 성과.
- Hard 인스턴스에서는 대부분 LLM이 Greedy도 아니고 uniform도 아닌 복잡한 행동을 보임.
- 보조실험 (단일 라운드 t에서 arm 선택) 수행:
  - 데이터 출처(Data source): 균등 무작위(Unif), UCB, TS 기반 히스토리
  - 통계: GreedyFrac (현재까지 가장 좋은 arm 선택 비율), LeastFrac (가장 적게 선택된 arm 선택 비율)
- Figure 10: Gpt-3.5 구성 및 베이스라인의 라운드별 선택 통계 요약
- 실험 결과, 선택 성향은 데이터 출처에 크게 좌우되어 "LLM이 과도하게 Greedy인지 혹은 균등한지" 판단 어려움.
- 일부 LLM(예: BNSN0)은 너무 Greedy한 경향, BSRN0는 너무 균등한 경향 나타내나, 다수 구성은 베이스라인과 유사 범위 내에 있음.
- 따라서, 장기적 실험에서는 LLM과 베이스라인 간 탐색 실패 유형과 성능 차이가 크지만, 단일 라운드 결정 기반 평가로는 실패 원인 규명에 한계가 있음.

---

# 4 Related work

- 본 논문은 LLMs(대형 언어 모델)의 능력을 이해하고자 하는 최근 연구 흐름에 속함.  
  - LLMs의 전반적 지능(Bubeck et al., 2023), 인과 추론(Kıcıman et al., 2023), 수학적 추론(Cobbe et al., 2021), 계획 수립(Valmeekam et al., 2023), 조합성(Yu et al., 2023) 등 다양한 능력 연구가 있으나 본 논문 주제와는 다소 거리가 있음.

- 본 연구는 주로 **in-context learning**(문맥 내 학습) 능력에 초점을 맞춤.  
  - 관련 연구들은 이론적(Xie et al., 2021; Zhang et al., 2023a 등) 및 실험적(Garg et al., 2022; Kirsch et al., 2022 등) 접근법으로 수행되었음.  
  - 대부분은 in-context **supervised learning**에 집중했으며, in-context **reinforcement learning**(RL)은 상대적으로 적은 관심을 받음.

- in-context reinforcement learning 관련 연구  
  - Laskin et al. (2022), Lee et al. (2023a), Raparthy et al. (2023) 등은 RL 알고리즘이나 전문가의 궤적 데이터를 사용해 처음부터 훈련한 모델을 대상으로 함.  
  - 이론적으로 Lee et al. (2023a), Lin et al. (2023) 등은 베이지안 메타강화학습 관점에서 기존 transformer가 Thompson sampling, upper confidence bounds (UCB) 같은 탐험 전략을 수행할 수 있음을 증명함.  
  - 그러나 이들은 LLM 사전학습 단계 개입이 필요하고, 기존 LLM이 표준 훈련 조건에서 탐험 능력을 지니는지는 다루지 않음.

- 본 논문과 가장 유사한 연구는 Coda-Forno et al. (2023)  
  - Gpt-3.5 기반 2-armed bandit 과제에서 in-context learning 성능을 평가.  
  - Greedy(최대 수익 arms 선택)와 유사하거나 약간 낮은 성능을 보였으나, UCB와 같은 복잡한 알고리즘과 비교할 충분한 시간 범위를 고려하지 않음.

- LLM을 실제 의사결정 문제에 적용하는 연구도 급증 중  
  - 게임, 프로그래밍, 의료 분야에 대한 연구(Shinn et al., 2023; Wang et al., 2023 등).  
  - Park et al. (2023)은 오픈 월드 환경에서 인간 행동을 시뮬레이션하는 생성 에이전트를 개발.  
  - Ahn et al. (2022), Xu et al. (2023)은 LLM을 탑재한 로봇 개발.

- 동시 수행된 관련 연구  
  - Wu et al. (2024): 쉬운 bandit 문제(2 arms, gap $$\Delta = 0.6$$)에서 GPT-4가 빠르게 최적 arm 선택, 인간과 비교, 단일 프롬프트 사용. 본 논문의 어려운 MAB 문제와 실험 결과는 차이가 있으나 쉬운 문제에서는 유사한 성공을 보임.  
  - Park et al. (2024): 주로 적대적 환경과 짧은 시간 범위 ($$T=25$$ bandit) 연구, 중요도 가중 손실(importance-weighted losses) 적용.  
    - 뒤이어 발표된 업데이트판에서는 긴 시간 범위 $$T=100$$ 에서도 LLM 성능 평가, 중요도 가중 처리 유무가 탐험 행동에 큰 영향(중요도 가중 시 성공, 제거 시 실패 증가).  
    - 이와 같은 결과는 본 논문에서 제시한 이력 사전처리(요약 또는 중요도 가중)가 LLM의 탐험적 행동을 끌어내는 데 중요하다는 결론과 일치함.

- 그 밖에 Schubert et al., Hayes et al., Coda-Forno et al. 등의 연구는 LLM이 결정과정에서 인간과 유사한 편향을 보이는지 탐구.  
- 최근 LLM 기반 강화학습 기법 리뷰: Cao et al. (2024).

- 후속 연구  
  - Monea et al. (2024), Nie et al. (2024)는 본 논문 결과를 확장하며 contextual bandits 및 vanilla MAB을 대상으로 LLM이 탐험에 실패함을 확인.  
  - 두 연구 모두 LLM 탐험 능력 향상을 위한 중재책을 제안:  
    - Monea et al.: interaction history를 균등하게 subsample하는 training-free 개입.  
    - Nie et al.: few-shot prompting 및 최적 시범 이용한 파인튜닝 포함.  
  - 제안된 중재책이 성능 개선을 주지만, 여전히 전통적 알고리즘과 경쟁하기에는 부족함.

---

# 4.1 Further background on multi-armed bandits

- 멀티 암 밴딧 문제와 본 논문에서 사용된 기본 알고리즘들에 대한 추가 배경 설명.
- 자세한 내용은 Bubeck and Cesa-Bianchi (2012), Slivkins (2019), Lattimore and Szepesvári (2020) 참고.

- **UCB 알고리즘 (Upper Confidence Bound, Auer et al., 2002a)**  
  - 각 암 $$a$$에 대해 인덱스 계산:  
    $$ \text{index}_a = \text{평균 보상}_a + \sqrt{\frac{C}{n_a}} $$  
    여기서 $$C = \Theta(\log T)$$이고, $$n_a$$는 해당 암을 지금까지 선택한 횟수.  
  - 각 라운드에서 가장 큰 인덱스를 갖는 암 선택.  
  - 보너스 항은 "불확실성 하에서의 낙관주의" 원칙을 구현함.  
  - 본 논문에서는 경험적으로 좋은 성능을 보이는 휴리스틱 $$C = 1$$로 설정한 UCB 버전을 사용.

- **Thompson Sampling**  
  - 베이지안 prior에서 암들의 평균 보상들이 초기 추출된 것으로 가정하고 시작.  
  - 매 라운드마다 지금까지의 히스토리를 바탕으로 posterior 계산 후, posterior에서 샘플링.  
  - 이 샘플을 실제 평균 보상으로 가정하고 가장 큰 평균 보상을 가진 암 선택.  
  - 본 설정에서 prior는 알고리즘의 파라미터이며, 각 암의 평균 보상을 $$[0, 1]$$ 구간에서 독립적이고 균일분포로 샘플링하는 표준 prior 사용.  
  - 각 암은 Beta-Bernoulli 공액 사전분포로 독립적으로 업데이트됨.  
  - 이 알고리즘은 근사 최적 후회(regret) 경계 및 좋은 실험적 성능을 가짐 (Kaufmann et al., 2012; Agrawal and Goyal, 2012, 2017).

- **알고리즘 성능 및 후회(Regret)**  
  - 후회는 최적 암의 기대 총 보상과 알고리즘의 기대 총 보상의 차이로 표현됨.  
  - UCB, Thompson Sampling 모두 다음과 같은 후회 상한을 가짐:  
    $$ O(\sqrt{K T \log T}) $$  
    이는 $$T$$와 $$K$$에 대해 거의 minimax 최적임.  
  - 또한, 고려하는 인스턴스에 대해 다음과 같은 인스턴스 최적 후회율도 가짐:  
    $$ O\left(\frac{K}{\Delta} \log T \right) $$

- **ε-Greedy 알고리즘**  
  - 성능 좋은 암 쪽으로 적응적으로 탐색을 유도하지 못해 비효율적임.  
  - 최적의 ε 값($$\epsilon \sim T^{-1/3}$$)에도 후회의 스케일은  
    $$ T^{2/3} $$  
  - ε를 고정하면 쉬운 인스턴스에서도 후회가 개선되지 않음.

- **Greedy 알고리즘**  
  - 전혀 탐색하지 않음.  
  - 초기 샘플링이 각 암당 $$n=1$$일 때, 좋은 암이 보상 0, 다른 암 중 하나가 보상 1을 반환하면 suffix failure 발생 가능.  
  - 이는 작은 $$n$$의 인위적 문제 아님; 모든 $$n$$에 대해 확률이  
    $$ \Omega\left(\frac{1}{\sqrt{n}}\right) $$  
    로 스케일되는 suffix failure가 발생 가능 (Banihashem et al., 2023).

---

# 5 Discussion and open questions

- 현재 대형 언어 모델(LLM)은 추가 개입 없이는 매우 기본적인 통계적 강화학습 및 의사결정 문제에서 요구되는 탐색(exploration)을 견고하게 수행하지 못하는 것으로 보임.

- 이에 대한 향후 연구 방향과 개입 방안을 아래와 같이 제안.

## 기본 개입과 방법론적 진전의 필요성

- 본 연구의 부정적 결과를 고려할 때, 다음과 같은 개입들이 유망할 수 있음:
  1. 프롬프트 변경 실험: 프롬프트 템플릿에 작은 변화를 주어 성능 개선 가능성을 탐색. 다만, 프롬프트 설계에 대한 민감성은 여전히 문제임.
  2. Few-shot 프롬프트 실험: 탐색 행동 예시를 포함하는 프롬프트 사용, 이를 이용해 LLM 미세조정 또는 학습 말뭉치에 탐색 행동 예시 추가.
  3. 계산기 등 보조 도구 사용 교육: 기본 산술 계산기나 확률 분포 샘플링을 위한 “랜덤라이저” 도구 사용법을 LLM에 학습시키기.

- 그러나 비용, 모델 접근성, 컴퓨팅 자원 문제가 큰 장벽임. 특히 긴 시간 지평선 $$T$$과 많은 반복 실험 수 $$N$$로 통계적 의미 있는 결과를 얻어야 하기 때문.

- 따라서 비용 효율적인 LLM-에이전트 행동 진단 및 이해를 위한 방법론적·통계적 진전(예: 대리 통계량 사용)이 필수적임.

## 복잡한 의사결정 문제에 대한 시사점

- 본 연구에서 사용한 간단한 다중 슬롯머신(MAB) 문제는 LLM의 탐색 행동과 개입 효과를 평가하기 위한 명확하고 통제 가능한 실험 장치임.

- MAB 환경에서의 탐색 실패는 복잡한 강화학습(RL) 및 의사결정 문제에서도 유사한 실패가 발생할 가능성을 시사함.

- 다만, MAB에서 성공한 해결책이 복잡한 환경에서는 잘 적용되지 않을 수 있으므로 주의가 필요함.

- 예를 들어, GPT-4가 요약된 상호작용 기록과 강화된 연쇄 추론(reinforced CoT)으로 MAB에서 성공적으로 탐색하였으나, 맥락이 있는 대규모 고차원 관찰(contextual bandits)에서는 외부에서 어떻게 기록을 요약해야 할지 불분명함(주석 1 참조).

- 선형 맥락 밴딧(linear contextual bandits) 환경에서도, 외부에서 선형회귀를 수행 후 프롬프트에 포함하는 등의 상당한 알고리즘적 개입 없이는 본 방법이 적용되기 어려움.

- 따라서 LLM이 의사결정 에이전트로서 얼마나 기능할 수 있는지 이해하기 위해선 더 깊은 알고리즘 개입에 대한 연구가 필수적임.