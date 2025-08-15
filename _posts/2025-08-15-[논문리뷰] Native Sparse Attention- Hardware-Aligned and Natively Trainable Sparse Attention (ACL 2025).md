---
title: "[논문리뷰] Native Sparse Attention- Hardware-Aligned and Natively Trainable Sparse Attention (ACL 2025)"
date: 2025-08-15 18:11:08 +0900
categories:
  - Paper Review
tags:
  - ACL 2025
  - NLP
---

NSA는 알고리즘·하드웨어 최적화를 결합한 natively trainable 희소 어텐션으로, 토큰을 거칠게 압축한 뒤 세밀히 선택하는 동적 계층적 전략으로 전역 맥락과 국소 정밀도를 동시에 보존합니다. 연산 집약도 균형 설계와 엔드투엔드 학습으로 사전학습 비용을 줄이면서도 Full Attention 대비 동등하거나 우수한 성능을 유지하고(64k 시퀀스에서 디코딩·순전파·역전파 모두 속도 향상) 효율을 크게 개선했습니다.

---

# 1. Introduction

- 배경 및 필요성
  - 차세대 대형 언어모델에서 긴 문맥(long-context) 모델링은 심층 추론, 저장소 수준 코드 생성, 다중 턴 자율 에이전트 등 실세계 응용에서 중요성이 커지고 있음.
  - OpenAI o-series, DeepSeek-R1, Gemini 1.5 Pro 등 최근 모델들은 수만 토큰 길이의 문서·코드베이스 처리, 수천 토큰에 걸친 일관된 대화 유지, 장거리 종속성에 대한 복잡한 추론을 가능하게 함.
- 문제점
  - 원래의(vanilla) Attention(softmax) 메커니즘은 계산 복잡도와 메모리 접근으로 인해 시퀀스 길이가 늘어날수록 지연(latency) 병목을 초래함.
  - 이론적 추정에 따르면, 64k 길이 디코딩 시 softmax 기반 attention 연산이 전체 지연의 약 70–80%를 차지함.
  - Attention 연산(예: softmax 기반)은 다음과 같이 기술될 수 있음:
    $$
    \text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
    $$
    그리고 계산 비용은 시퀀스 길이 n 및 차원 d에 대해 대략적으로
    $$
    \text{Cost}\propto n^2 d
    $$
- 기존 접근법 및 한계
  - 소프트맥스 어텐션의 본질적 희소성(sparsity)을 활용해 중요 쿼리-키 쌍만 선택하면 계산량을 크게 줄일 수 있음(관련 연구: KV-cache eviction, blockwise KV-cache selection, 샘플링/클러스터링/해싱 기반 방법 등).
  - 그러나 많은 기존 희소 어텐션 기법은 이론적 속도 이점이 실제 하드웨어 상의 속도 향상으로 잘 전환되지 않음.
  - 또한 대부분 방법이 학습(training) 단계에서 희소성 구조를 효과적으로 지원하지 못해 end-to-end 학습 효율성이 떨어짐.
- 핵심 도전 과제 (배포·학습 관점)
  - (1) 하드웨어 정렬(hardware-aligned) 추론 속도 향상: prefilling과 decoding 단계 모두에서 메모리 접근 및 하드웨어 스케줄링 병목을 완화하는 하드웨어 친화적 알고리즘 설계가 필요.
  - (2) 학습 인지(training-aware) 알고리즘 설계: 학습 비용을 줄이면서 성능을 유지할 수 있도록 학습 가능한 연산자와 효율적 역전파(backward) 지원이 필요.
- 제안: NSA (Natively trainable Sparse Attention)
  - 계층적 토큰 모델링(hierarchical token modeling)을 통합한 네이티브 학습 가능 희소 어텐션 아키텍처 제안.
  - 각 쿼리에 대해 키·값을 시간적 블록으로 정리하고 세 가지 병렬 어텐션 경로로 처리:
    - 압축된(coarse-grained) 토큰을 이용한 Compressed Attention (거친 패턴 포착)
    - 중요 블록을 선택하는 Selected Attention (상세 정보 보존)
    - 지역 문맥을 위한 Sliding Attention (로컬 정보)
  - 구조적 희소성 덕분에 많은 쿼리-키 점수 계산을 건너뛸 수 있음(그림에서 녹색 영역은 계산 필요 부분, 흰색은 생략 가능 부분).
- 시스템·학습 설계 혁신
  - 하드웨어 정렬 시스템: Tensor Core 활용과 메모리 접근을 고려한 블록단위 희소 어텐션 최적화로 산술집약도(arithmetic intensity) 균형 맞춤.
  - 학습 인지 설계: 안정적인 end-to-end 학습을 위한 효율적 알고리즘 및 backward 연산자 제공.
  - 커널 최적화(예: Triton 기반)로 실제 배포·학습에서 효율성 극대화.
- 실험 및 결과 요약
  - 27B 파라미터 변형기(transformer) 백본을 260B 토큰으로 프리트레이닝하여 일반 언어 평가, 긴 문맥 평가, 연쇄사고(chain-of-thought) 추론에서 평가.
  - NSA는 전반적 평가에서 Full Attention과 동등하거나 우수한 성능을 보였음.
  - A100 GPU 상에서 커널 속도 비교 결과, NSA는 디코딩·순전파·역전파 모든 단계에서 Full Attention 대비 실질적 속도 향상을 달성함(예: 64k 시퀀스 처리 시 디코딩·forward·backward 속도비 약 11.6×, 9.0×, 6.0×).
  - 긴 시퀀스에서 속도 향상 비율이 더 커지는 경향을 보였고, 계층적 희소 설계가 성능과 효율성 사이의 균형을 효과적으로 달성함.

---

# 2. Rethinking Sparse Attention Methods

- 개요
  - 최신 sparse attention는 이론적 계산복잡도를 크게 낮춤.
  - 그러나 대부분 방법이 추론 단계에서만 희소성을 적용하고, 사전학습된 Full Attention 백본을 유지하여 구조적 편향(architectural bias)을 남김. 이로 인해 희소성의 실질적 장점을 온전히 활용하지 못함.
  - 본절에서는 두 가지 관점(효율적 추론의 환상, 학습 가능한 희소성의 신화)으로 기존 접근의 한계를 체계적으로 분석함.

- 2.1 효율적인 추론의 환상
  - 계산 상 희소성을 달성해도 실제 추론 지연(latency)이 줄지 않는 주요 원인:
    - 단계(phase)-제한적 희소성
      - H2O 등은 오토회귀 디코딩에서만 희소성을 적용하나, prefilling 단계에서 주의맵 계산·인덱스 빌드 등 비용이 큰 전처리를 요구.
      - 반면 MInference는 prefilling만을 타깃으로 함.
      - 결과적으로 적어도 한 단계는 Full Attention에 준하는 비용을 요구하여 전체 가속 효과가 제한됨(예: prefilling 중심 작업: 도서 요약, 코드 완성 / decoding 중심 작업: 긴 생각 연쇄 reasoning).
    - 고급 어텐션 아키텍처와의 비호환성
      - MQA( Multiple-Query Attention )와 GQA( Grouped-Query Attention )는 디코딩 시 KV를 여러 쿼리 헤드와 공유하여 메모리 접근 병목을 줄였음.
      - Quest 등 방법은 각 헤드가 독립적으로 KV-캐시의 부분집합을 선택함. GQA 기반 모델에서는 같은 그룹 내 모든 헤드의 선택의 합집합이 실제 KV 접근량을 결정함:
        $$
        \text{KV-access}_{\text{GQA group}} = \bigcup_{h\in G} S_h
        $$
      - 따라서 계산량은 줄어들더라도 KV-캐시의 메모리 접근량이 여전히 높게 남아 실질적 속도 향상을 제한함.
  - 요약: 많은 기존 방법은 KV-캐시 감소나 이론적 연산 감소에 집중하지만, 고급 프레임워크/백엔드에서는 지연 감소를 이루지 못함. 이는 아키텍처·하드웨어 친화적 구현을 결합한 알고리즘 설계의 필요성을 제기함.

- 2.2 학습 가능한 희소성의 신화
  - 인퍼런스 전용 접근에서 얻은 두 가지 핵심 인사이트:
    - (1) 성능 저하 위험
      - 사후적으로 희소성을 적용하면 모델이 사전학습된 최적화 경로에서 벗어나게 됨. Chen et al.에 따르면 상위 20%의 attention이 전체 attention 점수의 약 70%만 커버하여(상위 일부가 나머지를 완전히 대체하지 못함), 검색 헤드 등 사전학습된 구조가 추론 시 가지치기에 취약해짐:
        $$
        \text{Top}_{20\%}\ \text{attention} \approx 70\% \ \text{of total}
        $$
    - (2) 학습 효율성 요구
      - 장시퀀스 학습(더 긴 문서로의 사전학습, long-context fine-tuning, 강화학습 등)은 현대 LLM 개발에 필수적.
      - 그러나 기존 희소 어텐션은 주로 추론을 목표로 하여 학습 단계의 계산 문제를 충분히 해결하지 못함. 이는 장문 컨텍스트 모델의 효율적 학습을 저해함.
  - 기존 sparse를 학습에 맞추려는 시도에서 드러난 추가 문제:
    - 비학습 가능(Non-trainable) 컴포넌트
      - ClusterKV( k-means ), MagicPIG( SimHash 기반 선택 ) 등은 이산적 연산을 포함하여 계산 그래프에 불연속성을 만들고, 토큰 선택 과정으로의 그래디언트 흐름을 차단함.
    - 비효율적 역전파
      - HashAttention 같은 토큰-그레인 선택 방식은 어텐션 계산 시 KV 캐시에서 많은 개별 토큰을 불러와야 하며, 이 비연속적 메모리 접근은 FlashAttention 같은 고속 블록 연산(연속 메모리 접근)을 활용하지 못하게 함.
      - 결과적으로 구현은 낮은 하드웨어 활용도로 떨어지며 학습 효율이 크게 저하됨.

- 2.3 네이티브 희소성의 필요성
  - 위의 추론 효율성 및 학습 가능성 한계는 희소 어텐션의 근본적 재설계를 요구함.
  - 저자들은 계산 효율성과 학습 요건을 동시에 만족하는 네이티브 희소 어텐션 프레임워크 NSA를 제안함.
  - 이후 섹션에서 NSA의 알고리즘 설계와 연산자(operator) 구현 세부를 설명함.

---

# 3. Methodology

- 개요
  - 알고리즘 설계와 하드웨어 친화적 커널 최적화를 결합한 접근.
  - 배경(어텐션, 산술 강도), 전체 프레임워크(NSA), 알고리즘 구성요소(압축·선택·슬라이딩 윈도우), 그리고 Triton 기반 하드웨어 최적화 커널 설계를 순차적으로 제시.

- 배경
  - 어텐션(캐주얼 self-attention): 각 쿼리 q_t가 이전 모든 키·값에 대한 가중합을 계산.
    $$o_t = \mathrm{Attn}(q_t, k_{:t}, v_{:t})$$
    $$\mathrm{Attn}(q_t, k_{:t}, v_{:t}) = \frac{\sum_{i=1}^{t} \alpha_{t,i} v_i}{\sum_{j=1}^{t} \alpha_{t,j}},\quad \alpha_{t,i} = \exp\!\left(\frac{q_t^\top k_i}{\sqrt{d_k}}\right)$$
  - 산술 강도(arithmetic intensity): 연산 대비 메모리 접근 비율. GPU별 임계값 이상이면 연산(bound by FLOPS), 이하이면 메모리 대역폭(bound).
  - 단계별 특성:
    - 학습(prefill)·트레이닝: 배치 매트릭스 연산으로 연산 집약적(연산-bound).
    - 오토리그레시브 디코딩: 매 스텝마다 전체 KV 캐시를 로드하므로 메모리 대역폭 제약(메모리-bound).
  - 따라서 최적화 목표가 단계별로 다름(학습에서는 연산 절감, 디코딩에서는 메모리 접근 절감).

- 전체 프레임워크 (NSA)
  - 기본 아이디어: 원래의 전체 키·값 k_{:t}, v_{:t}을 쿼리별로 더 작고 정보 밀집된 재표현 ˜K_t, ˜V_t로 동적으로 재매핑하여 어텐션을 수행.
    $$\tilde K_t = f_K(q_t, k_{:t}, v_{:t}),\quad \tilde V_t = f_V(q_t, k_{:t}, v_{:t})$$
    $$o^*_t = \mathrm{Attn}(q_t, \tilde K_t, \tilde V_t)$$
  - 여러 매핑 전략(c ∈ C)을 설계하여 병렬 결합:
    $$o^*_t = \sum_{c\in C} g^c_t \cdot \mathrm{Attn}(q_t, \tilde K^c_t, \tilde V^c_t)$$
    - C = {cmp, slc, win} (압축, 선택, 슬라이딩 윈도우)
    - g^c_t ∈ [0,1]: 각 분기(branch) 가중치(MLP + sigmoid로 산출).
  - 재매핑된 키·값의 총 개수 N_t:
    $$N_t = \sum_{c\in C} \mathrm{size}[\tilde K^c_t]$$
    - 목표: 높은 희소성 유지, 즉 $$N_t \ll t$$

- 알고리즘 설계 (f_K, f_V의 세 가지 전략)
  - 1) 토큰 압축 (Token Compression)
    - 연속 블록들의 키·값을 블록-레벨 표현으로 집계하여 압축된 키·값 생성.
    - 정의(블록 길이 l, stride d, intra-block 포지션 인코딩 및 MLP φ):
      $$\tilde K_{cmp,t} = f^{K}_{cmp}(k_{:t}) = \Big[\phi\big(k_{id+1 : id+l}\big)\Big]_{0\le i \le \lfloor\frac{t-l}{d}\rfloor}$$
    - 결과: 더 거친(granular) 표현으로 연산량 감소. 유사하게 ˜V_{cmp,t} 정의.

  - 2) 토큰 선택 (Token Selection)
    - 압축만으로 손실되는 세부 정보를 보존하기 위해 중요 토큰(블록)을 선택.
    - 블록 단위 선택의 이유:
      - GPU에서 연속 블록 접근이 랜덤 접근보다 효율적(메모리·텐서코어 활용).
      - 어텐션 점수는 공간적 연속성을 가지는 경향(이웃 블록이 유사한 중요도).
    - 블록 중요도 계산: 압축 토큰에 대한 중간 어텐션 점수를 재활용.
      $$p^{cmp}_t = \mathrm{Softmax}\big(q_t^\top \tilde K_{cmp,t}\big)$$
      - p^{cmp}_t는 압축 키들에 대한 q_t의 attention 분포.
    - 블록 스키마가 다를 때(압축 블록 l, 선택 블록 l′, d|l, d|l′, l \le l′) 중요도 집계:
      $$p^{slc}_t[j] = \sum_{m=0}^{\frac{l'}{d}-1}\sum_{n=0}^{\frac{l}{d}-1} p^{cmp}_t\Big[\tfrac{l'}{d}\,j - m - n\Big]$$
    - 헤드 그룹(GQA/MQA)의 경우, 같은 그룹 내 헤드들 간에 일관된 블록 선택을 위해 헤드별 점수를 합산:
      $$p^{slc,'}_t = \sum_{h=1}^{H} p^{slc,(h)}_t$$
    - Top-n_{sparse} 블록 선택:
      $$I_t = \{\, i \ \vert\ \mathrm{rank}\big(p^{slc,'}_t[i]\big) \le n \,\}$$
      $$\tilde K_{slc,t} = \mathrm{Cat}\big(\{\, k_{i l' + 1 : (i+1)l'} \ \vert\ i\in I_t \,\}\big)$$
    - 선택된 키·값( ˜K_{slc,t}, ˜V_{slc,t})을 쿼리와 함께 어텐션에 사용.

  - 3) 슬라이딩 윈도우 (Sliding Window)
    - 국소(local) 패턴은 빠르게 학습되어 다른 분기들이 국소 패턴에 '쇼트컷' 당하는 것을 방지하기 위해 별도 분기 할당.
    - 최근 w 토큰을 윈도우로 유지:
      $$\tilde K_{win,t} = k_{t-w:t},\quad \tilde V_{win,t} = v_{t-w:t}$$
    - 세 분기(압축·선택·윈도우)의 키·값을 독립적으로 유지하여 분기 간 그래디언트 간섭을 줄임.
    - 최종 출력은 게이팅된 합(앞의 $$o^*_t$$ 식)으로 결합.

- 커널 설계 (Triton 기반 하드웨어 최적화)
  - 목표: 학습·prefill 단계에서 FlashAttention 수준의 속도 달성, 디코딩 측면에서는 KV 캐시 공유 아키텍처(GQA/MQA) 중심으로 최적화.
  - 압축·윈도우 분기는 기존 FlashAttention-2와 호환되며, 선택(sparse selection) 분기를 위한 전용 커널을 설계함.
  - 핵심 최적화 아이디어: 쿼리 블록 단위가 아니라 GQA 그룹 단위로 쿼리 헤드들을 SRAM에 로드(같은 그룹은 동일한 희소 KV 블록을 공유).
  - 커널 구조의 주요 특성:
    - 그룹 중심 데이터 로딩:
      - 각 내부 루프에서 그룹 내 모든 헤드의 쿼리 Q(그룹, d_k)를 위치 t에 대해 로드하고, 공유되는 희소 KV 블록 인덱스 I_t를 가져옴.
    - 공유 KV 페칭:
      - 내부 루프에서 I_t로 지정된 연속적인 키/값 블록을 SRAM으로 순차 로드:
        - K ∈ R^{B_k × d_k}, V ∈ R^{B_k × d_v} (B_k는 커널 블록 크기, B_k \vert l′)
      - 중복 KV 전송을 줄여 메모리 접근 최소화.
    - 외부 루프를 Triton의 grid에 배치:
      - 선택 블록 수 n이 쿼리 블록별로 거의 일정하므로, 쿼리/출력 루프를 grid 스케줄러에 맡겨 단순화 및 최적화.
  - 기대 효과:
    - 그룹-단위 공유로 중복 KV 로드 제거.
    - SRAM에서의 블록 단위 계산으로 산술 강도 향상 및 SM(Streaming Multiprocessor) 간 워크로드 균형화.
    - 결과적으로 학습 단계에서 실무적인 속도 개선(FlashAttention 수준의 효율) 도달.

- 요약(핵심 포인트)
  - NSA는 압축·선택·슬라이딩 윈도우의 세 분기로 KV를 재매핑하여 희소하면서 정보 밀집된 표현으로 어텐션을 계산.
  - 블록 단위 선택과 압축의 조합으로 GPU 친화적 접근(연속 메모리 접근, 텐서코어 활용)을 보장.
  - Triton 기반 그룹 중심 커널로 KV 전송을 줄이고 산술 강도를 높여 실무적 성능을 확보.

---

# 4. Experiments

- 개요
  - NSA를 세 가지 관점에서 평가: (1) 일반 벤치마크 성능, (2) 장문(long-context) 벤치마크 성능, (3) chain-of-thought 추론 성능.
  - 비교 대상: Full Attention baseline 및 최신 sparse attention 방법들(H2O, infLLM, Quest, Exact-Top).
  - 효율성 분석은 섹션 5에 별도 기술(학습/추론 속도 등).

- 4.1 Pretraining Setup
  - 모델 아키텍처
    - Grouped-Query Attention (GQA) + Mixture-of-Experts (MoE) 백본.
    - 총 파라미터: 27B, 활성 파라미터: 3B.
    - 레이어 수: 30, 히든 차원: 2560.
    - GQA 설정: 그룹 수 = 4, 총 어텐션 헤드 = 64.
    - 헤드별 차원:
      - $$d_q = d_k = 192$$
      - $$d_v = 128$$
    - MoE: DeepSeekMoE 구조(72 routed experts, 2 shared experts), top-k 전문가 수 = 6.
    - 안정성: 첫 레이어의 MoE는 SwiGLU MLP로 대체.
  - 학습 데이터/절차
    - 양자 모델(Full Attention과 NSA)은 8k 길이 텍스트로 270B 토큰 사전학습.
    - 그 후 YaRN을 사용해 32k 길이 텍스트로 장문 적응을 위한 continued training 및 supervised fine-tuning 수행.
    - 두 모델 모두 완전 수렴까지 학습하여 공정 비교 보장.
  - NSA 하이퍼파라미터(압축/선택 관련)
    - $$l = 32,\ d = 16,\ l' = 64,\ n = 16,\ w = 512$$
      - (참고: n에는 1개의 초기 고정 블록과 2개의 로컬 블록이 포함됨)
  - 수렴 및 손실
    - 사전학습 손실(그림 4): NSA가 Full Attention보다 낮은 손실을 보이며 안정적으로 수렴.

- 4.2 Baselines Methods
  - 비교 대상 sparse attention 방법들: H2O, infLLM, Quest, Exact-Top (Exact-Top: full attention 점수 계산 후 각 쿼리에 대해 top-n 키 선택).
  - 장문 평가 시 모든 sparse 방법의 활성 토큰 수(희소도)는 동일하게 설정(공정 비교).
  - 일반 평가(대부분 샘플이 로컬 윈도우 내 길이)에서는 sparse 방법들이 Full Attention과 유사하게 동작하므로 주요 비교는 NSA vs Full Attention으로 제시.
  - Chain-of-thought 평가는 장문 SFT가 필요하므로 sparse attention baselines는 훈련을 지원하지 않아 Full Attention과만 비교.

- 4.3 Performance Comparison
  - 일반 평가(knowledge, reasoning, coding)
    - 벤치마크: MMLU, MMLU-PRO, CMMLU, BBH, GSM8K, MATH, DROP, MBPP, HumanEval.
    - 결과 요약 (Table 1)
      - NSA는 희소성에도 불구하고 전반적으로 우수: 9개 메트릭 중 7개에서 Full Attention보다 상회.
      - 특히 reasoning 관련 향상 두드러짐: DROP +0.042, GSM8K +0.034.
      - 해석: 사전학습된 희소 어텐션이 중요 정보에 집중하게 하여 잡음 제거 및 추론 성능 향상에 기여.
  - 장문 평가(Long-Context)
    - Needle-in-a-haystack (64k 컨텍스트): NSA가 모든 위치에서 완벽한 검색 정확도 달성(계층적 희소 어텐션 덕분).
      - 설계 철학: 압축 토큰으로 저비용 전역 스캔 → 선택 토큰으로 정밀한 지역 정보 복원(전역 인식 + 지역 정밀성 유지).
    - LongBench 평가(표 2)
      - 비교 대상: H2O, infLLM, Quest, Exact-Top, Full Attn.
      - 모든 sparse 방법의 활성 토큰 예산을 2560으로 동일 설정(이 중 선두 128 토큰 + 512 로컬 토큰 포함).
      - 평균 점수: NSA = 0.469, Full Attention = 0.437, Exact-Top = 0.423.
        - NSA는 Full Attention 대비 +0.032, Exact-Top 대비 +0.046 향상.
      - 세부 개선 예: multi-hop QA(HPQ, 2Wiki)에서 각각 +0.087, +0.051, 코드 이해(LCC) +0.069, Passage retrieval(PassR-en) +0.075.
      - 원인 분석: (1) 사전학습 단계에서의 네이티브 희소 패턴 최적화로 모듈-모델 동기화 가능, (2) 계층적 설계로 지역/전역 균형 달성.
  - Chain-of-Thought Reasoning 평가
    - 설정
      - DeepSeek-R1로부터 지식 증류, 32k 길이의 수학적 추론 트레이스 10B 토큰으로 supervised fine-tuning 수행.
      - 생성: 샘플링 온도 0.7, top-p = 0.95, 각 문제당 16개 응답의 평균 점수 사용.
      - 생성 컨텍스트 한계 실험: 8k 및 16k 토큰.
    - 결과 (Table 3)
      - Full Attention-R: 8k = 0.046, 16k = 0.092
      - NSA-R: 8k = 0.121, 16k = 0.146
      - 개선폭: 8k에서 +0.075, 16k에서 +0.054 (NSA-R 우세)
    - 해석
      - 네이티브 희소 어텐션 패턴이 장거리 논리적 의존성 포착에 유리.
      - 하드웨어 정렬된 설계로 충분한 컨텍스트 밀도를 유지하여 깊은 추론 체인 지원.
  - 연산/커널 성능(요약)
    - Triton 기반 NSA 커널 vs FlashAttention-2 (그림 6)
      - Forward 시간 속도향상: 약 2.1× (8k), 3.8× (16k), 6.3× (32k), 9.0× (64k).
      - Backward 시간 속도향상: 약 1.1× (8k), 2.0× (16k), 3.4× (32k), 6.0× (64k).
      - 결론: 입력 길이가 길수록 NSA 커널의 대기시간 이점이 더 뚜렷.

- 종합 결론(Section 4 요약)
  - NSA는 네이티브 희소 어텐션 설계와 계층적 압축·선택 메커니즘 덕분에:
    - 일반 벤치마크에서 Full Attention과 대등하거나 우수한 성능을 보이고,
    - 장문 과제에서 뛰어난 검색/추론 능력을 발휘하며,
    - chain-of-thought 수학 추론에서도 유의미한 성능 향상을 달성.
  - 또한 구현 수준에서 긴 입력에 대해 계산·메모리 이점(추론·학습 커널 속도 향상)을 확인함.

---

# 5. Efficiency Analysis

- 실험 환경
  - 하드웨어: 8-GPU A100 시스템
  - 모델 설정:
    - $$g = 4$$ (GQA 그룹 수)
    - $$h = 16$$ (그룹당 헤드 수)
    - $$d_k = 192$$ (쿼리/키 차원)
    - $$d_v = 128$$ (값 차원)
  - NSA 관련 하이퍼파라미터:
    - $$l = 32$$ (compression block size)
    - $$d = 16$$ (sliding stride)
    - $$l' = 64$$ (selected block size)
    - $$n = 16$$ (selected block count)
    - $$w = 512$$ (sliding window size)

- 5.1 Training Speed
  - 비교 대상: Triton 기반 구현의 NSA vs Full Attention (및 Triton 기반 FlashAttention-2) — 동일 백엔드로 공정 비교.
  - 결과 요약:
    - 컨텍스트 길이가 길어질수록 NSA의 속도 이점이 커짐.
    - 최대 향상: 순전파(Forward)에서 $$9.0\times$$, 역전파(Backward)에서 $$6.0\times$$ (컨텍스트 길이 64k 기준).
    - 예상 속도비(컨텍스트 길이별): $$4\times,\,6.4\times,\,9.1\times,\,11.6\times$$ (8k, 16k, 32k, 64k).
  - 속도 향상의 주요 원인:
    - Blockwise 메모리 접근 패턴으로 Tensor Core 활용 극대화(연속적(coalesced) 로드).
    - 커널 내 정교한 루프 스케줄링으로 KV 중복 전송 제거.

- 5.2 Decoding Speed
  - 디코딩은 저연산 강도(low arithmetic intensity)·메모리 바운드 작업으로, KV 캐시 로딩량이 속도 결정 요인임.
  - NSA의 디코딩 시 최대 로드 토큰 수:
    - 압축 토큰: $$\left\lceil\frac{s - l}{d}\right\rceil$$ (여기서 $$s$$는 캐시된 시퀀스 길이)
    - 선택된 토큰: $$n\,l'$$
    - 이웃 토큰: $$w$$
  - 결과 및 해석:
    - 로딩되는 메모리 액세스량이 줄어들어 디코딩 지연이 크게 감소.
    - 컨텍스트 길이 증가 시 이점이 커지며, 최대 $$11.6\times$$ 속도 향상(64k 기준).
    - 메모리 액세스 볼륨과 지연은 거의 선형 관계를 보임(표 4 참조).

---

# 6. Discussion

- 개요
  - NSA 개발 과정과 다양한 sparse attention 전략을 탐색하면서 얻은 인사이트와 한계점을 정리.
  - 대안 전략에서 발생한 문제들을 분석하고, attention 패턴 시각화를 통해 설계 동기를 설명.

- 6.1 대체 토큰 선택 전략에서의 문제점
  - Key-클러스터링 기반 전략의 한계
    - 클러스터링 기반(예: ClusterKV)의 장점에도 불구하고 실제 적용 시 세 가지 주요 병목 발생:
      - 동적 클러스터링으로 인한 비자명한 계산 오버헤드.
      - 연산자 최적화 어려움 및 클러스터 간 불균형 문제(특히 MoE에서 Expert Parallelism(EP) 그룹 실행 시간의 편향으로 인한 지속적 부하 불균형).
      - 정기적 재클러스터링과 청크-순차(chunk-sequential) 학습 요구로 인한 구현 제약.
    - 이러한 요인들이 결합되어 실환경 배포에 큰 제약을 초래.

  - 블록 단위 선택(Quest, InfLLM 등)의 문제
    - 블록 중요도 점수를 계산해 $$\text{top-}n$$ 블록을 선택하는 방법은 두 가지 핵심 문제를 가짐:
      - 선택 연산이 비미분 가능하므로, 신경망 기반 중요도 예측은 보조 손실(auxiliary loss)을 필요로 하며 연산자 오버헤드를 증가시키고 종종 성능 저하를 초래.
      - 휴리스틱(파라미터 없는) 중요도 계산은 재현률(recall)이 낮아 성능이 떨어짐.
    - 이들 방법을 3B 파라미터 모델에서 NSA 및 Full Attention과 비교 실험한 결과(그림 참고) 보조손실 기반/휴리스틱 기반 모두 낮은 성능을 보였음.

  - 보조손실 기반 중요도 추정 세부사항
    - 각 토큰에 추가 쿼리 및 각 블록에 대표 키를 도입.
    - 블록별 감독 신호는 블록 내 attention 점수의 평균 풀링으로 생성하고, KL 발산으로 중요도 예측을 감독.
    - 디코딩 효율을 위해 블록 평균 쿼리가 아닌 개별 쿼리 그레인(개별 토큰 쿼리)을 유지.
    - 이 접근은 SeerAttention과 개념적으로 유사하지만, 본 실험에서는 성능이 낮음.

  - 휴리스틱(파라미터 없음) 구현 및 cold-start
    - Quest 전략을 따라 쿼리와 키 청크의 좌표별 min-max의 곱으로 직접 선택을 구현.
    - 초기 1000 스텝은 Full Attention으로 학습한 뒤 전환하는 cold-start도 시도했으나 성능 개선 미미.
    - 결과적으로 NSA가 더 나은 손실 곡선을 보임(3B 모델 기준).

- 6.2 시각화(Attention 분포 관찰)
  - 27B Full Attention 사전학습 모델의 attention 맵 시각화 결과:
    - attention 점수들이 블록 단위(clustered)로 뭉치는 경향을 보이며, 인접한 키들이 유사한 attention 값을 가지는 패턴 관찰.
    - 이러한 블록화(blockwise clustering) 현상은 시퀀스 상 인접 토큰들이 쿼리 토큰과 공유하는 의미적 관계가 있을 가능성을 시사.
  - 설계적 시사점
    - 연속된 토큰 블록을 선택하는 방식은 계산 효율을 높이면서도 고(高)-attention 패턴을 잘 보존할 수 있다는 아이디어를 뒷받침.
    - NSA가 개별 토큰이 아니라 연속 블록 단위로 동작하도록 설계된 배경이 이러한 관찰에서 비롯됨.
  - 향후 과제
    - 인접 토큰 간의 정확한 의미적 관계 및 블록화 원인에 대한 추가 분석 필요.
    - 블록 기반 sparse 메커니즘의 일반화 가능성과 실환경 적용성 개선 연구 권장.

---

# 7. Related Works

- 개요
  - 어텐션 연산의 효율화를 위한 희소화(sparse attention) 방법들을 검토.
  - 핵심 전략별로 세 그룹으로 분류: (1) 고정 희소 패턴, (2) 동적 토큰 프루닝, (3) 쿼리-인식 선택.
  - 각 그룹별 대표적 연구들을 소개하고, 본문(우리의 NSA)과의 차별점을 제시.

- 7.1. Fixed Sparse Pattern (고정 희소 패턴)
  - SlidingWindow: 쿼리가 고정된 윈도우 내에서만 어텐션을 계산하도록 하는 전형적 접근.
  - StreamingLLM (Xiao et al., 2023): attention sinks와 로컬 윈도우를 결합해 연속 텍스트 스트림 처리.
  - MoA (Fu et al., 2024a), DuoAttention (Xiao et al., 2024b): 장문 시퀀스 모델링을 위해 로컬 정보와 sink 정보 사용.
  - Longformer (Beltagy et al., 2020): 로컬 윈도우 어텐션과 글로벌 토큰을 교차 배치하여 장문 처리.
  - 본 연구의 차이: NSA는 사전 정의된 희소 패턴에 의존하지 않고 패턴을 자동으로 학습하여 전체 문맥 활용 가능성을 열어둠.

- 7.2. Dynamic Token Pruning (동적 토큰 프루닝)
  - 목표: 디코딩 시 메모리와 연산 비용을 줄이기 위해 KV-cache를 동적으로 줄임.
  - H2O (Zhang et al., 2023b), BUZZ (Zhao et al., 2024), SepLLM (Chen et al., 2024a): 디코딩 중 중요하지 않은 토큰을 동적으로 제거하여 KV-cache 사용량 감소.
  - FastGen (Ge et al., 2023), HeadKV (Fu et al., 2024b): 개별 어텐션 헤드별로 다른 전략을 적용해 계산 최적화.
  - SnapKV (Li et al., 2024): 가장 중요한 특징만 선택적으로 유지해 KV-cache를 축소.
  - 본 연구의 차이: NSA는 추론 시의 프루닝 중심 방법들과 달리 학습 단계에서부터 희소성을 내재화함.

- 7.3. Query-Aware Selection (쿼리-인식 선택)
  - 목표: 쿼리 의존적 토큰 선택으로 어텐션 연산을 줄이면서 품질 유지.
  - Quest (Tang et al., 2024): 블록 단위 선택, 쿼리와 키 청크의 좌표별 min-max 곱으로 청크 중요도 추정.
  - InfLLM (Xiao et al., 2024a): attention sinks, 로컬 컨텍스트, 검색 가능한 청크를 결합해 대표 키를 선택하여 청크 중요도 추정.
  - HashAttention (Desai et al., 2024): 쿼리와 키를 학습된 함수로 Hamming 공간에 매핑해 핵심 토큰 식별을 추천 문제로 모델링.
  - ClusterKV (Liu et al., 2024): 키를 군집화한 뒤 쿼리-클러스터 유사도로 관련 클러스터 선택.
  - MInference (Jiang et al., 2024), TokenSelect (Wu et al., 2024): 토큰 수준 중요도 스코어링 기반으로 KV 쌍 선택.
  - SeerAttention (Gao et al., 2024): 쿼리와 키를 공간 블록으로 분리하고 블록 단위 선택 수행.
  - 본 연구의 차이: NSA는 학습·프리필링·디코딩을 포함한 전체 모델 생애주기에서 하드웨어 친화적(hardware-aligned) 희소 어텐션 연산을 달성.

---

# 8. Conclusion

- NSA는 하드웨어에 정렬된(sparse) 어텐션 아키텍처로, 장기 문맥(long-context) 모델링에서 효율성을 목표로 설계됨  
- 계층적 토큰 압축(hierarchical token compression)과 블록별 토큰 선택(blockwise token selection)을 학습 가능한 구조 내에 통합하여 구현됨  
- 이러한 설계로 학습과 추론 모두에서 가속을 달성하면서도 Full Attention 수준의 성능을 유지함  
- 계산 복잡도 관점에서 기존 전역 어텐션의 비용이 $$O(n^2)$$인 반면, NSA는 압축된 토큰 수 m에 의해 효과적으로 $$O(n \cdot m)\quad(m \ll n)$$ 수준으로 비용을 낮춤  
- 벤치마크에서 Full-Attention 기반 기준선과 동등한 전반적 성능을 보였고, 장기 문맥 평가에서는 오히려 더 우수한 모델링 능력을 보였음  
- 추론 및 추론 관련 작업에서 추론 지연(latency)과 계산량이 측정 가능하게 감소하였고, 실질적인 속도 향상(speedup)을 달성함  
- 종합적으로 NSA는 하드웨어 친화적 희소 어텐션 설계로서 장기 문맥 처리와 추론 효율성, 그리고 추론 능력 향상 간의 균형을 성공적으로 제공함