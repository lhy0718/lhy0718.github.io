---
title: "[논문리뷰] Do LLM Agents Have Regret? A Case Study in Online Learning and Games (ICLR 2021)"
date: 2025-05-26 14:31:04 +0900
categories:
  - Paper Review
tags:
  - ICLR 2021
  - LLM in Game Theory
---

LLM 기반 에이전트가 반복 게임 및 온라인 학습 상황에서 **후회(regret)** 를 줄이는 방향으로 학습 또는 추론할 수 있는지를 실증적으로 검증하고, 실패하는 경우를 탐색하며, 이를 개선할 수 있는 새로운 훈련 기법(regret-loss)을 제안함.

---

## 1. 서론

- LLM은 점점 더 다양한 의사결정 상황에 투입되고 있음.
- 특히 멀티에이전트 환경에서의 상호작용(전형적 게임이론 상황)은 실제 애플리케이션에서 매우 중요함.
- 본 연구는 LLM의 **의사결정 품질을 "후회" 메트릭**으로 정량적으로 평가함.

---

## 2. 배경 및 개념

- **온라인 학습과 반복 게임**: 시간에 따라 변하는 환경에서 점진적 의사결정을 내리는 상황.
- **Regret**: 후회란 이상적인 전략(사후적으로 가장 잘했을 전략) 대비 성능 손실을 의미함.
- **Transformer 및 Self-Attention 구조**: LLM 내부 구조 개요 제시.

---

## 3. 실험: 사전학습된 LLM의 후회 행동

### 3.1 실험 설계
- LLM에게 이전 라운드의 보상 벡터(혹은 손실 히스토리)를 주고, 다음 정책을 추론하게 하는 프롬프트 제공.
- 명시적으로 "후회 최소화"를 요구하지 않고, 히스토리를 기반으로 추론만 요구.

### 3.2 온라인 학습 결과
- GPT-4, GPT-4 Turbo, GPT-3.5, Llama-3-70B 등 다양한 모델이 포함됨.
- 대부분의 비확률적 시나리오(예: 가우시안 손실, 선형 트렌드, 사인 트렌드)에서 **서브리니어(sublinear) regret**을 달성함.

### 3.3 멀티에이전트 반복 게임 결과
- payoff matrix는 직접 제공되지 않고, 보상 벡터로만 결정됨.
- 다수의 일반-합 게임에서 LLM은 **서브리니어 regret**을 보임.
- 다수의 에이전트가 상호작용할수록 예측 난이도 증가로 성능 저하 발생.

### 3.4 실패 사례
- 고전적인 FTL(Follow-the-Leader) 알고리즘이 실패하는 상황(Alternating Loss 등)에서 GPT-4도 **선형 regret**을 보임.
- 특히 적대적인 loss 벡터가 등장할 때 GPT-4는 큰 후회를 가짐.

---

## 4. 왜 사전학습 LLM은 (후회를) 갖거나 갖지 않는가?

- **Quantal Response** 이론 등 인간의 결정을 모사한 모델을 사용해 LLM의 의사결정을 설명하고자 시도.
- 특정 가정 하에서 사전학습된 LLM은 **FTPL(Follow-the-Perturbed-Leader)** 알고리즘과 유사한 행동을 보일 수 있음.

---

## 5. 새로운 방법: Regret-Loss 도입

### 5.1 Regret-Loss 정의
- 라벨 없이 후회 최소화를 유도하는 **비지도 학습 손실 함수**.
- 기존의 교사강화 학습(supervised loss)과 달리, 최적 행동의 정답이 필요 없음.

### 5.2 이론적 보장
- **통계적 일반화 정리** 및 **최적화 보장** 제시.
- 단층 self-attention 구조 하에서는 **FTRL(Follow-the-Regularized-Leader)** 알고리즘에 수렴 가능.

### 5.3 Regret-Loss로 학습된 Transformer
- 단일/다층 Transformer 모두 Regret-Loss로 학습 시 No-regret 행동을 보임.

### 5.4 실험 결과
- GPT-4 기반 모델이 실패하던 시나리오에서도, Regret-Loss로 학습된 Transformer는 성능 향상.

---

## 6. 결론 및 향후 연구

- LLM은 온라인 및 전략적 의사결정 상황에서 일정 수준의 **합리적(no-regret) 행동**을 보일 수 있음.
- 그러나 특정 상황에서는 여전히 후회(regret)를 보임 → 이에 대한 개선이 필요함.
- 향후 연구 방향:
  - 다양한 regret 정의(external-regret, swap-regret, policy regret 등) 적용.
  - 사회적 효율성 관점에서 LLM 평가.
  - 복잡한 멀티에이전트 환경 및 무한 반복 게임으로의 확장.
  - Regret-loss 및 프롬프트 설계 기법 개발.

---

## 📚 참고 문헌 일부
- Cesa-Bianchi & Lugosi (2006), Shalev-Shwartz (2012), Roughgarden (2015)
- Brookins & DeBacker (2023), Fan et al. (2023): LLM 반복 게임 기반 분석
- Fu et al. (2023), Du et al. (2023): 협상/토론 게임 기반 상호작용 증진
- Zhao et al. (2023) “CompeteAI”: LLM 경쟁 행동 분석