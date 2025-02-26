---
title: "[Paper Review] Graph of Thoughts: Solving Elaborate Problems with Large Language Models (AAAI 2024)"
date: 2025-02-26 16:00:00 +0900
categories:
  - Paper Review
tags:
  - LLM
  - NLP
  - AAAI
  - Graph of Thoughts
---

**Summary:** This paper introduces the **Graph of Thoughts (GoT)** framework, which enhances the prompt capabilities of large language models (LLMs) by modeling information in graph form, creating synergies, and improving performance across a variety of tasks. GoT can extend new thought transformations, contributing to innovative prompting approaches.

---

## **1. Introduction**

Large language models (LLMs) play a crucial role in AI. In recent years, transformer models, such as GPT, PaLM, and LLaMA, have rapidly evolved. Prompt engineering is an effective method for solving LLM tasks, where task descriptions are fed to the LLM as input. A well-structured prompt allows the LLM to generate text using an autoregressive token mechanism to solve the problem. These prompts may include example tasks and solutions (few-shot prompting) or may contain none at all (zero-shot prompting).

<img alt="image" src="https://github.com/user-attachments/assets/4e691799-31fd-4081-bc80-8b1bac7443ac" />

<img alt="image" src="https://github.com/user-attachments/assets/f94122f3-8939-4f45-9e7a-38a7ba8dd616" />

The **Chain-of-Thought (CoT)** approach enhances performance by including intermediate steps in problem-solving. An advanced form of CoT, **Self-Consistency with CoT (CoT-SC)**, generates multiple CoTs and selects the best result. More recently, the **Tree of Thoughts (ToT)** approach has been proposed, which allows modeling multiple paths of thought. However, ToT restricts the thought process to a rigid tree structure, inherently limiting cognitive flexibility.

This study proposes modeling the LLM's thought process as an arbitrary graph structure, allowing for more powerful prompting. Human thinking, unlike simple chains of thought, often forms complex networks. We introduce **Graph of Thoughts (GoT)**, where the nodes represent thoughts, and edges represent dependencies between them. GoT integrates CoT and ToT to support more complex thought patterns and solves multiple design challenges.

GoT allows fine-grained control over independent thoughts and can incorporate new thought transformations and reasoning patterns. We present several use cases of GoT (e.g., sorting, keyword counting, set operations) and explain the implementation of the graph-based paradigm in detail. GoT improves performance by over 70% while reducing costs by over 31%.

Finally, we propose a new metric, **"Volume of Thought,"** for evaluating prompting strategies, demonstrating that GoT inherently supports larger volumes of thought compared to other approaches.

---

## **2. The GoT Framework**

<img alt="image" src="https://github.com/user-attachments/assets/106f0c18-548e-401c-8f19-adbfbaa33912" />

**GoT (Graph of Thoughts)** is formed based on a conversation consisting of a user message (prompt) and the LLM's response (thought). GoT is modeled as a tuple (G; T; E; R), where:

- **G** represents the LLM's reasoning process,
- **T** represents potential thought transformations,
- **E** is an evaluation function for scoring thoughts,
- **R** is a ranking function to select relevant thoughts.

### **2.1 Reasoning Process**

The reasoning process is modeled as a directed graph **G = (V; E)**, where **V** is the set of vertices, and **E** is the set of edges. Each vertex contains a solution to the problem, and a directed edge **(t1; t2)** indicates that **t2** is derived from **t1**. Thought transformations can evolve **G**, such as merging the highest-scoring thoughts into new thoughts.

### **2.2 Thought Transformation**

GoT, due to its graph-based structure, enables new thought transformations. For example, multiple input articles can be merged into a coherent summary, or several sorted subarrays can be merged into one final sorted array. Each transformation is modeled as **T(G; pθ)**, where **G** reflects the current state of reasoning. Thoughts can also be explicitly removed if necessary.

### **2.3 Scoring and Ranking Thoughts**

Thoughts are scored to understand whether the current solution is good enough. The scoring function is modeled as **E(v; G; pθ)**. GoT can rank thoughts using the ranking function **R(G; pθ; h)**, returning the top-scoring thought. For example, in sorting, the score may correspond to the number of correctly sorted elements.

---

## **3. System Architecture & Extensibility**

The GoT architecture consists of interacting modules, including:

- **Prompter**: Prepares the prompt to send to the LLM, handling the graph structure's encoding details. Users can implement custom graph encodings for specific purposes.
- **Parser**: Extracts information from the LLM's thoughts to construct the thought state, which is updated in the **Graph Reasoning State (GRS)**.
- **Scoring & Validation**: Verifies if the LLM's thoughts meet accuracy requirements and assigns scores. The scoring can be provided either by the LLM or humans.
- **Controller**: Selects thoughts in the GRS and determines which transformations to apply, passing them to the **Prompter**. The controller decides whether to finish the process or initiate further interaction based on the execution plan.

The **Graph of Operations (GoO)** defines the static structure for task decomposition, while **GRS** maintains information about the LLM's reasoning process (e.g., current tasks, states of generated thoughts, scores). These elements provide an extensible API for easily implementing various prompt formats, as shown in Figure 2 (green section).

---

## **4. Example Use Cases**

<img alt="image" src="https://github.com/user-attachments/assets/ff3711d3-3196-485a-89f9-31c93cc52b74" />

This section covers two example use cases: **sorting** and **set intersection**.

- **Sorting**: Sorting a sequence of numbers from 0 to 9. LLMs fail to consistently match the number of duplicates according to sequence length. GoT uses a merge-sort algorithm by breaking the input sequence into subarrays, sorting each subarray individually, and merging them to produce the final result.

- **Set Intersection**: Finding the intersection of two sets, similar to sorting, involves partitioning the second input set into subsets and using the LLM to determine the intersection with the first set before aggregating the final intersection set.

Additionally, **keyword counting** is addressed, where GoT divides the input text into several segments, counts the frequency of keywords in each segment, and aggregates the results. The final score is calculated by summing the absolute differences between the total number of keywords and the accurate count.

In the **document merging** case, GoT generates a new **Non-Disclosure Agreement (NDA)** document by minimizing redundancy and maximizing information retention from multiple input documents. LLMs request two values to evaluate the result’s redundancy and information retention, using the harmonic mean to finalize the values.

---

## **5. The Latency-Volume Tradeoff**

GoT (Generative of Thoughts) demonstrates improvements in the tradeoff between latency and volume compared to previous prompting methods. **Volume** is defined as the number of prior LLM thoughts that can influence a given thought **t**. Formally, volume is the number of thoughts with a path leading to **t**.

The total cost for all prompting methods is fixed at **O(n)**, assuming the time for generating a single thought is **O(1)**. The structures of various prompting methods are as follows:

- **CoT**: A set of independent chains, originating from a single starting thought.
- **CoT-SC**: Similar to CoT but splits into **k** paths to reduce latency.
- **ToT**: A complete **k-ary** tree structure.
- **GoT**: A complete **k-ary** tree combined with identical-sized "mirror" trees at each leaf.

A table comparing the latency and volume of different prompting methods:

| Prompting Method | Latency (Latency) | Volume (Volume) |
| ---------------- | ----------------- | --------------- |
| CoT              | N                 | N               |
| CoT-SC           | N/k               | N/k             |
| ToT              | logkN             | O(logkN)        |
| GoT              | logkN             | N               |

GoT is the only approach that simultaneously offers **low latency** and **high volume**, enabled by the utilization of a set of thoughts that allow reaching the final thought from any intermediate thought in the graph decomposition.

---

## **6. Evaluation**

We demonstrate the advantages of GoT by comparing it with existing state-of-the-art methods. GoT consistently outperforms ToT, and experiments were also conducted comparing GoT with IO, CoT, and CoT-SC. In our evaluation methodology, we used 100 input samples for each task and comparison criterion, set the temperature to 1.0, and applied a 4k context size. To maintain similar costs across experiments, the number of thoughts for each technique was fixed.

GoT shows higher performance in all problem instances compared to ToT and ToT2, with a cost reduction when compared to ToT. For example, at P = 128, GoT reduces the median error by 62% compared to ToT, while achieving over 31% cost savings. This advantage arises from GoT’s characteristic of breaking down complex tasks into simpler subtasks and independently solving them before progressively merging them into the final result.

GoT consistently provides higher quality results compared to IO and CoT. For instance, in the sorting problem at P = 64, GoT’s median error is 65% lower than CoT and 83% lower than IO. As the problem size (P) increases, the advantages of GoT become more pronounced, with the median error decreasing as the problem size grows.

When tasks are broken down into subtasks, the size of the response and the input (in terms of tokens) decrease proportionally to the degree of decomposition. However, the “static” part of the prompt (e.g., a few examples) can introduce significant overhead. Ultimately, the goal of graph decomposition is to allow the LLM to solve the task accurately with a single prompt for most cases, significantly reducing the number of subsequent correction steps.

---

## **7. Related Work**

Below is a summary of research related to GoT:

**Prompt Paradigms and Approaches**  
Various studies related to prompting exist, as detailed in Section 1 and Table 1. These include Plan-and-Solve (Wang et al., 2023a), Fu et al. (2022), self-taught reasoner (Zelikman et al., 2022), Shum et al. (2023), automated prompt generation (Shin et al., 2020; Li and Liang, 2021; Lester et al., 2021), simultaneous extension into concise bullet-point answers (Ning et al., 2023), and optimal prompt selection from candidate sets (Zhou et al., 2022). Most of these can be expressed through GoT abstractions.

**Prompt Chaining**  
Prompt chaining involves linking different LLMs together (Creswell et al., 2022; Nye et al., 2021; Wu et al., 2022; Dohan et al., 2022; Qiao et al., 2023; Wu et al., 2022). GoT can be extended to serve as an execution engine for such methods.

**Self-Reflection and Self-Evaluation**  
Recent studies have introduced self-reflection and self-evaluation (Shinn et al., 2023; Paul et al., 2023; Madaan et al., 2023; Xie et al., 2023; Zhu et al., 2023). GoT partially relies on self-evaluation when expanding the graph of thoughts within the prompt.

**LLMs and Planning**  
There has been considerable research on planning complex tasks with LLMs (Huang et al., 2022a,b; Zhang et al., 2023; Yao et al., 2023b; Yang et al., 2023; Wang et al., 2023c). GoT provides a paradigm for generating complex graph-based plans and can serve as a general framework to enhance such approaches.

**Graphs and Graph Computing**  
Graphs have become an essential and popular element in general computing environments (Lumsdaine et al., 2007; Malewicz et al., 2010; Gregor and Lumsdaine, 2005a,b; Sakr et al., 2021). Recently, there has been growing interest in graph databases (Robinson et al., 2015; Besta et al., 2022b, 2023b,d,c), graph pattern matching (Fan et al., 2010; Cheng et al., 2008; Teixeira et al., 2015; Besta et al., 2021a,b, 2022d), graph streaming (Feng et al., 2015; Dhulipala et al., 2019; Besta et al., 2023a), graph machine learning, and graph neural networks (Hamilton et al., 2017; Wu et al., 2021; Zhou et al., 2020; Zhang et al., 2022; Chami et al., 2020; Bronstein et al., 2017; Besta et al., 2022a,c; Gianinazzi et al., 2021; Scarselli et al., 2008). This study uses graph abstractions as a key mechanism to enhance the prompting capabilities of LLMs.

---

## **8. Conclusion**

Prompt engineering is one of the central fields in large language model (LLM) research, enabling efficient use of LLMs without the need for model updates. However, designing effective prompts is a challenging task. In this study, we propose a new paradigm, **Graph of Thoughts (GoT)**, which effectively solves a variety of tasks. GoT models the reasoning of LLMs as a graph, with thoughts as nodes and dependencies between thoughts as edges. This structure reflects a non-linear problem-solving process where sets of thoughts or intermediate solutions are combined.

GoT outperforms various prompting methods, achieving 62% better sorting quality compared to ToT, while reducing costs by more than 31%. We also introduce a new metric, **“Volume of Thought,”** which represents the range of output information in GoT and further demonstrates its superiority. Graph abstraction has been the foundation of computer and AI design for decades, and this study applies this concept to the field of prompt engineering.

---

## **Reader's Opinions**

- The methodology of extending ToT to perform reasoning exploration in the form of a graph allows LLMs to handle more complex "thoughts."
- However, the necessity of solving mathematical tasks like sorting, keyword counting, and set operations using LLMs is not immediately clear.
- For such mathematical calculations, using an LLM agent with external tools may be a more effective and efficient approach.
