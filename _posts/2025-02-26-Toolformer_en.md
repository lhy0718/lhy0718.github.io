---
title: "[Paper Review] Toolformer: Language models can teach themselves to use tools (NeurIPS 2023)"
date: 2025-02-26 00:00:00 +0900
categories:
  - Paper Review
tags:
  - LLM
  - NLP
  - NeurIPS
---

**Summary:** This paper introduces **Toolformer**, a model that learns to enhance task performance by utilizing external tools through a **self-supervised approach**. Toolformer significantly improves **zero-shot performance** by leveraging various APIs and demonstrates the ability to compete with larger models.

---

# 1 Introduction

Large language models (LLMs) like those discussed by Brown et al. (2020) and Chowdhery et al. (2022) have demonstrated impressive capabilities in zero-shot and few-shot natural language processing tasks. However, these models exhibit several limitations, including the inability to access real-time information, tendencies to hallucinate facts, difficulties with low-resource languages, inadequate mathematical skills, and a lack of awareness of temporal progression.

To address these challenges, a potential solution is providing LLMs with the ability to use external tools, such as search engines and calculators. Current methodologies, however, often rely on extensive human annotations or restrict tool usage to specific tasks, limiting broader applicability.

To overcome these shortcomings, we introduce Toolformer, a novel model designed to learn tool use in a self-supervised manner without requiring large human annotations. Toolformer aims to maintain the model's generality and autonomy in deciding when and how to utilize different tools.

Our approach leverages recent advancements in in-context learning, enabling LLMs to generate comprehensive datasets from minimal examples of tool usage. By annotating a language modeling dataset with potential API calls and utilizing a self-supervised loss to evaluate their effectiveness, Toolformer can learn to control various tools autonomously.

Through experimental validation, Toolformer, based on a pretrained 6.7 billion parameter GPT-J model, shows significant improvements in zero-shot performance across diverse downstream tasks, outperforming larger models such as GPT-3 and various other baselines.

---

# 2 Approach

Our goal is to enhance a language model $ M $ with the capability to utilize various tools via API calls, ensuring that the inputs and outputs for each API can be represented as text sequences. This facilitates seamless integration of API calls into any text using special tokens to delineate the start and end of each call. An API call is represented as a tuple $ c = (a_c, i_c) $, where $ a_c $ is the API name and $ i_c $ is the input.

The linearization of API calls is defined as:

- $ e(c) = <API> a_c(i_c) </API> $
- $ e(c,r) = <API> a_c(i_c) \rightarrow r </API> $

The approach consists of converting a dataset $ C $ of plain texts into an augmented dataset $ C^\* $ with API calls through three primary steps:

1. **Sampling API Calls**: We create prompts that encourage the language model $ M $ to suggest API calls for each text example in $ C $. We sample potential positions for API calls based on the probability assigned by $ M $ and filter them using a defined threshold.

2. **Executing API Calls**: Once generated, the API calls are executed to retrieve results, which are obtained as text sequences.

3. **Filtering API Calls**: We evaluate the usefulness of each API call and its result by comparing the model's prediction loss with and without the API call and its output. Only API calls that significantly reduce the loss are retained.

After filtering, the remaining API calls are merged with the original texts, and the new dataset $ C^\* $ is used to fine-tune $ M $. This fine-tuning occurs while maintaining the original content, allowing the language model to learn how to effectively use tools based on feedback.

During inference, $ M $ continues regular decoding until producing a token indicating the expectation of an API call response. At this point, we interrupt decoding to retrieve the response and continue thereafter.

---

# 3 Tools

We explore various tools to address the limitations of regular language models (LMs). These tools meet two main criteria: (i) their inputs and outputs can be represented as text sequences, and (ii) we can obtain a few demonstrations of their intended use. The five tools discussed are:

1. **Question Answering System**: Utilizing Atlas, a retrieval-augmented LM fine-tuned on Natural Questions, this tool answers simple fact-based questions.

2. **Calculator**: This tool performs simple arithmetic operations (addition, subtraction, multiplication, and division) with results rounded to two decimal places.

3. **Wikipedia Search Engine**: A search engine that returns short text snippets from Wikipedia based on search terms. It uses a BM25 retriever to extract relevant information, offering a more comprehensive understanding of topics compared to the question answering tool.

4. **Machine Translation System**: A multilingual translation model, NLLB, translates phrases into English from various languages (up to 200, including low-resource ones). The source language detection is managed through a fastText classifier.

5. **Calendar API**: This tool returns the current date upon query, providing necessary temporal context for time-sensitive predictions.

Further details on each tool can be found in Appendix A.

---

# 4 Experiments

In this section, we explore the effectiveness of our model's ability to utilize various tools autonomously for different tasks without additional supervision. Our evaluation is divided into several subsections, including experimental setup, downstream tasks, language modeling performance, and the impact of model size on tool usage.

## 4.1 Experimental Setup

- **Dataset Generation**: We use a subset of CCNet as our language modeling dataset and GPT-J as our language model. Heuristics are applied to filter texts for specific APIs to ensure utility. Statistics on examples with API calls are provided.
- **Model Finetuning**: The language model is fine-tuned with a specific batch size and learning rate.
- **Baseline Models**: Several models are compared, including a standard GPT-J model, a version fine-tuned on CCNet without API calls, and one fine-tuned with augmented API calls.

## 4.2 Downstream Tasks

Evaluations are conducted in zero-shot settings across various tasks:

- **LAMA**: The model shows significant performance improvements in completing statements with required facts, outperforming other baseline models.
- **Math Datasets**: The Toolformer demonstrates superior mathematical reasoning abilities across multiple benchmarks, utilizing the calculator tool extensively.
- **Question Answering**: Most question-answering tasks show Toolformer effectively leveraging the Wikipedia search tool, but it still trails behind larger models like GPT-3.
- **Multilingual Question Answering**: Toolformer's multilingual capabilities are assessed, showing improvements with API usage. However, performance varies significantly across languages.
- **Temporal Datasets**: The model outperformed baselines on temporal-related tasks, although improvements were not solely attributed to the calendar tool due to restrictions on simultaneous API calls.

## 4.3 Language Modeling

We verify the model’s language modeling capabilities using WikiText and CCNet datasets. Findings indicate that finetuning with API calls does not degrade the language model performance significantly.

## 4.4 Scaling Laws

An investigation into the effect of model size indicates that the ability to effectively use external tools manifests around the 775M parameter mark, with smaller models not showing significant improvements. As models increase in size, their capacity to leverage tools and excel at tasks also increases, underscoring a notable gap in prediction performance with and without tool usage.

Overall, this section establishes the utility and effectiveness of our approach in various contexts, confirming that tool usage enhances performance while maintaining foundational language modeling abilities.

---

# 5 Analysis

## Decoding Strategy

This section analyzes the modified decoding strategy introduced in a prior section. Instead of consistently generating the most likely token, the strategy allows for generating an `<API>` token if it falls within the top k likely tokens. Performance metrics were evaluated on the T-REx subset of LAMA and WebQS as k varied. As k increases, the model utilizes API calls more frequently, with performance improving in instances where API calls are made. Particularly, the model displayed better calibration when performing API calls, as it selected them for poor-performing cases, although this calibration diminished at higher values of k.

## Data Quality

The analysis extends to the quality of API calls produced by the model. A qualitative examination of instance examples reveals that advantageous API calls generally correspond with high scores from the metric used for filtering. Conversely, lower scores often suggest less useful API calls, albeit some instances with lower scores still reduce perplexity without providing valuable information. Overall, noise in non-filtered API calls can contribute positively by preventing the model from overly adhering to the outcomes of every API call made.

---

# 6 Related Work

## Language Model Pretraining

Various strategies enhance language models during pretraining by integrating additional textual information, such as:

- Metadata (Keskar et al., 2019)
- HTML tags (Aghajanyan et al., 2021)
- Wikipedia markup (Schick et al., 2022)
- Related texts from information retrieval systems (Guu et al., 2020; Borgeaud et al., 2021; Izacard et al., 2022)

Unlike these methods that provide additional information indiscriminately, Toolformer autonomously determines when to solicit relevant information.

## Tool Use

Several efforts have been made to enable language models to utilize external tools like:

- Search engines
- Web browsers
- Calculators
- Translation systems
- Python interpreters

These models typically learn tool usage in one of two ways:

1. Relying on extensive human supervision (Komeili et al., 2022; Nakano et al., 2021; Thoppilan et al., 2022)
2. Tailoring a few-shot set-up for specific tasks (Gao et al., 2022; Lazaridou et al., 2022; Yao et al., 2022)

Toolformer’s self-supervised training allows it to learn to use tools without needing specific prompts, differing from TALM (Parisi et al., 2022), which also employs a self-supervised approach but is limited to fine-tuning in downstream task settings.

## Bootstrapping

Self-training and bootstrapping methods have been explored in numerous areas, including:

- Word sense disambiguation (Yarowsky, 1995)
- Relation extraction (Brin, 1999; Agichtein and Gravano, 2000)
- Parsing (McClosky et al., 2006; Reichart and Rappoport, 2007)
- Sequence generation (He et al., 2020)
- Few-shot text classification (Schick and Schütze, 2021a)
- Retrieval (Izacard and Grave, 2021)
- Reasoning (Zelikman et al., 2022)

Similarly, Toolformer utilizes a perplexity-based filtering step during training on its predictions.

---

# 7 Limitations

Our method allows language models (LMs) to learn tool usage in a self-supervised manner, but there are notable limitations:

1. **Chained Tool Use**: Toolformer cannot use tools in a chain, as API calls for each tool are generated independently, lacking examples of chained tool usage in the finetuning dataset.

2. **Interactive Tool Usage**: The current approach does not support interactive use of tools like search engines, which require the ability to browse results or refine queries, limiting its applicability.

3. **Sensitivity to Input Wording**: Models trained with Toolformer show sensitivity to the exact phrasing of inputs when deciding to call an API, reflecting the general sensitivity of LMs to prompts.

4. **Sample Inefficiency**: The method can be highly sample-inefficient; for instance, processing over a million documents may yield only a few thousand successful API calls.

5. **Tool-Dependent Costs**: Toolformer does not currently consider the computational costs associated with making API calls, which varies by tool.

Addressing these limitations could enhance the efficacy and usability of the model.

---

# 8 Conclusion

We have presented Toolformer, a self-supervised language model capable of utilizing various tools like search engines, calculators, and translation systems through API calls. Toolformer is trained via fine-tuning on numerous sampled API calls filtered for perplexity reduction in future tokens. This approach significantly enhances the zero-shot performance of a 6.7B parameter GPT-J model, allowing it to surpass the performance of the larger GPT-3 model across various tasks.
